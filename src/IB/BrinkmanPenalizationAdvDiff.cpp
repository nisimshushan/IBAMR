// ---------------------------------------------------------------------
//
// Copyright (c) 2019 - 2019 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ibamr/AdvDiffHierarchyIntegrator.h"
#include "ibamr/BrinkmanPenalizationAdvDiff.h"
#include "ibamr/app_namespaces.h" // IWYU pragma: keep
#include "ibamr/ibamr_enums.h"

#include "ibtk/IndexUtilities.h"
#include "ibtk/ibtk_utilities.h"

#include "Box.h"
#include "CartesianPatchGeometry.h"
#include "CellData.h"
#include "CellVariable.h"
#include "HierarchyCellDataOpsReal.h"
#include "IntVector.h"
#include "Patch.h"
#include "PatchHierarchy.h"
#include "PatchLevel.h"
#include "RefineAlgorithm.h"
#include "RefineOperator.h"
#include "RefineSchedule.h"
#include "SideData.h"
#include "SideGeometry.h"
#include "SideIndex.h"
#include "Variable.h"
#include "VariableContext.h"
#include "VariableDatabase.h"
#include "tbox/Database.h"
#include "tbox/MathUtilities.h"
#include "tbox/Pointer.h"
#include "tbox/RestartManager.h"
#include "tbox/Utilities.h"

#include <cmath>
#include <string>
#include <utility>

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace IBAMR
{
/////////////////////////////// STATIC ///////////////////////////////////////

/////////////////////////////// PUBLIC //////////////////////////////////////
BrinkmanPenalizationAdvDiff::BrinkmanPenalizationAdvDiff(std::string object_name,
                                                         Pointer<AdvDiffHierarchyIntegrator> adv_diff_solver)
    : d_object_name(object_name), d_adv_diff_solver(adv_diff_solver)
{
    return;
} // BrinkmanPenalizationAdvDiff

void
BrinkmanPenalizationAdvDiff::setTimeInterval(double current_time, double new_time)
{
    d_current_time = current_time;
    d_new_time = new_time;
    return;
} // setTimeInterval

void
BrinkmanPenalizationAdvDiff::setBrinkmanCoefficient(double chi)
{
    // TODO: Allow for flexibility in setting chi for different Q and level sets?
    d_chi = chi;
    return;
} // setBrinkmanCoefficient

void
BrinkmanPenalizationAdvDiff::registerBrinkmanBoundaryCondition(Pointer<CellVariable<NDIM, double> > Q_var,
                                                               Pointer<CellVariable<NDIM, double> > ls_solid_var,
                                                               std::string bc_type,
                                                               double bc_val)
{
    auto bc_type_enum = string_to_enum<AdvDiffBrinkmanPenalizationBcType>(bc_type);
    if (bc_type_enum != DIRICHLET && bc_type_enum != NEUMANN)
    {
        TBOX_ERROR("BrinkmanPenalizationAdvDiff::registerBrinkmanBoundaryCondition\n"
                   << "  unsupported Brinkman penalization BC type: "
                   << enum_to_string<AdvDiffBrinkmanPenalizationBcType>(bc_type_enum) << " \n"
                   << "  valid choices are: DIRICHLET, NEUMANN\n");
    }

    // Store BC options within the map
    for (const auto& e : d_Q_bc)
    {
        // Ensuring that a solid level set is only registered with each transported quantity once
        if (e.first == Q_var)
        {
            for (const auto& tup : e.second)
            {
                if (std::get<0>(tup) == ls_solid_var)
                {
                    TBOX_ERROR("BrinkmanPenalizationAdvDiff::registerBrinkmanBoundaryCondition\n"
                               << "  two separate boundary conditions on level set variable " << ls_solid_var->getName()
                               << " \n"
                               << "  are being registered for transported quantity " << Q_var->getName() << "\n");
                }
            }
        }
    }
    auto bc_tuple = std::make_tuple(ls_solid_var, bc_type_enum, bc_val);
    d_Q_bc[Q_var].push_back(bc_tuple);
    return;
}

void
BrinkmanPenalizationAdvDiff::registerBrinkmanBoundaryCondition(
    Pointer<CellVariable<NDIM, double> > Q_var,
    std::vector<Pointer<CellVariable<NDIM, double> > > ls_solid_vars,
    std::vector<string> bc_types,
    std::vector<double> bc_vals)
{
    TBOX_ASSERT(ls_solid_vars.size() == bc_types.size() == bc_vals.size());
    for (int i = 0; i < ls_solid_vars.size(); ++i)
    {
        registerBrinkmanBoundaryCondition(Q_var, ls_solid_vars[i], bc_types[i], bc_vals[i]);
    }
    return;
}

void
BrinkmanPenalizationAdvDiff::computeBrinkmanOperator(int C_idx, Pointer<CellVariable<NDIM, double> > Q_var)
{
#if !defined(NDEBUG)
    TBOX_ASSERT(d_Q_bc.find(Q_var) != d_Q_bc.end());
#endif
    auto brinkman_zones = d_Q_bc[Q_var];
    for (const auto& bc_tup : brinkman_zones)
    {
        // Get the BC specifications for each zone
        Pointer<CellVariable<NDIM, double> > ls_solid_var = std::get<0>(bc_tup);
        AdvDiffBrinkmanPenalizationBcType bc_type = std::get<1>(bc_tup);
        double bc_val = std::get<2>(bc_tup);

        // Get the solid level set info
        VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
        const int ls_current_idx =
            var_db->mapVariableAndContextToIndex(ls_solid_var, d_adv_diff_solver->getCurrentContext());

        if (bc_type == DIRICHLET)
        {
            computeDirichletBrinkmanOperator(C_idx, ls_current_idx);
        }
        else if (bc_type == NEUMANN)
        {
            TBOX_ERROR("TODO");
        }
        else
        {
            // This statement should not be reached
            TBOX_ERROR("Error in BrinkmanPenalizationAdvDiff::computeBrinkmanOperator: \n"
                       << "bc_type = " << enum_to_string(bc_type) << "\n");
        }
    }
    return;
}

void
BrinkmanPenalizationAdvDiff::computeBrinkmanForcing(int C_idx, Pointer<CellVariable<NDIM, double> > Q_var)
{
#if !defined(NDEBUG)
    TBOX_ASSERT(d_Q_bc.find(Q_var) != d_Q_bc.end());
#endif
    auto brinkman_zones = d_Q_bc[Q_var];
    for (const auto& bc_tup : brinkman_zones)
    {
        // Get the BC specifications for each zone
        Pointer<CellVariable<NDIM, double> > ls_solid_var = std::get<0>(bc_tup);
        AdvDiffBrinkmanPenalizationBcType bc_type = std::get<1>(bc_tup);
        double bc_val = std::get<2>(bc_tup);

        // Get the solid level set info
        VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
        const int ls_current_idx =
            var_db->mapVariableAndContextToIndex(ls_solid_var, d_adv_diff_solver->getCurrentContext());

        if (bc_type == DIRICHLET)
        {
            computeDirichletBrinkmanForcing(C_idx, ls_current_idx, bc_val);
        }
        else if (bc_type == NEUMANN)
        {
            TBOX_ERROR("TODO");
        }
        else
        {
            // This statement should not be reached
            TBOX_ERROR("Error in BrinkmanPenalizationAdvDiff::computeBrinkmanForcing: \n"
                       << "bc_type = " << enum_to_string(bc_type) << "\n");
        }
    }
    return;
}

/////////////////////////////// PRIVATE //////////////////////////////////////
void
BrinkmanPenalizationAdvDiff::computeDirichletBrinkmanOperator(int C_idx, int phi_idx)
{
    // Compute kappa*(1-Hphi) throughout the hierarchy
    Pointer<PatchHierarchy<NDIM> > patch_hierarchy = d_adv_diff_solver->getPatchHierarchy();
    const int coarsest_ln = 0;
    const int finest_ln = patch_hierarchy->getFinestLevelNumber();
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM> > level = patch_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM> > patch = level->getPatch(p());
            const Pointer<CartesianPatchGeometry<NDIM> > patch_geom = patch->getPatchGeometry();
            const Box<NDIM>& patch_box = patch->getBox();
            const double* patch_dx = patch_geom->getDx();
            const double alpha = 2.0 * patch_dx[0]; // TODO: More general expression for width?

            Pointer<CellData<NDIM, double> > ls_solid_data = patch->getPatchData(phi_idx);
            Pointer<CellData<NDIM, double> > C_data = patch->getPatchData(C_idx);

            for (Box<NDIM>::Iterator it(patch_box); it; it++)
            {
                CellIndex<NDIM> ci(it());
                double phi = (*ls_solid_data)(ci);
                double Hphi;
                if (phi < -alpha)
                {
                    Hphi = 0.0;
                }
                else if (std::abs(phi) <= alpha)
                {
                    Hphi = 0.5 + 0.5 * phi / alpha + 1.0 / (2.0 * M_PI) * std::sin(M_PI * phi / alpha);
                }
                else
                {
                    Hphi = 1.0;
                }

                (*C_data)(ci) = (1.0 - Hphi) * d_chi;
            }
        }
    }
    return;
}

void
BrinkmanPenalizationAdvDiff::computeDirichletBrinkmanForcing(int C_idx, int phi_idx, double bc_val)
{
    // Compute kappa*(1-Hphi)*Q_bc throughout the hierarchy
    Pointer<PatchHierarchy<NDIM> > patch_hierarchy = d_adv_diff_solver->getPatchHierarchy();
    const int coarsest_ln = 0;
    const int finest_ln = patch_hierarchy->getFinestLevelNumber();
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM> > level = patch_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM> > patch = level->getPatch(p());
            const Pointer<CartesianPatchGeometry<NDIM> > patch_geom = patch->getPatchGeometry();
            const Box<NDIM>& patch_box = patch->getBox();
            const double* patch_dx = patch_geom->getDx();
            const double alpha = 2.0 * patch_dx[0]; // TODO: More general expression for width?

            Pointer<CellData<NDIM, double> > ls_solid_data = patch->getPatchData(phi_idx);
            Pointer<CellData<NDIM, double> > C_data = patch->getPatchData(C_idx);

            for (Box<NDIM>::Iterator it(patch_box); it; it++)
            {
                CellIndex<NDIM> ci(it());
                double phi = (*ls_solid_data)(ci);
                double Hphi;
                if (phi < -alpha)
                {
                    Hphi = 0.0;
                }
                else if (std::abs(phi) <= alpha)
                {
                    Hphi = 0.5 + 0.5 * phi / alpha + 1.0 / (2.0 * M_PI) * std::sin(M_PI * phi / alpha);
                }
                else
                {
                    Hphi = 1.0;
                }

                (*C_data)(ci) = (1.0 - Hphi) * d_chi * bc_val;
            }
        }
    }
    return;
}
/////////////////////////////// NAMESPACE ////////////////////////////////////

} // namespace IBAMR

//////////////////////////////////////////////////////////////////////////////
