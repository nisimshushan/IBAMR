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
namespace
{
static const int SIDEG = 1;
static const int NOGHOSTS = 0;
} // namespace
/////////////////////////////// PUBLIC //////////////////////////////////////
BrinkmanPenalizationAdvDiff::BrinkmanPenalizationAdvDiff(std::string object_name,
                                                         Pointer<AdvDiffHierarchyIntegrator> adv_diff_solver)
    : d_object_name(object_name), d_adv_diff_solver(adv_diff_solver)
{
    // Initialize variables
    VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
    const IntVector<NDIM> no_ghosts = NOGHOSTS;
    const IntVector<NDIM> side_ghosts = SIDEG;
    d_B_var = new SideVariable<NDIM, double>(d_object_name + "::");
    d_B_scratch_idx =
        var_db->registerVariableAndContext(d_B_var, var_db->getContext(d_object_name + "B::SCRATCH"), no_ghosts);
    d_B_chi_scratch_idx =
        var_db->registerVariableAndContext(d_B_var, var_db->getContext(d_object_name + "BCHI::SCRATCH"), no_ghosts);

    d_div_var = new CellVariable<NDIM, double>(d_object_name + "::DIV");
    d_div_B_scratch_idx =
        var_db->registerVariableAndContext(d_div_var, var_db->getContext(d_object_name + "B::SCRATCH"), no_ghosts);
    d_div_B_chi_scratch_idx =
        var_db->registerVariableAndContext(d_div_var, var_db->getContext(d_object_name + "BCHI::SCRATCH"), no_ghosts);
    d_chi_scratch_idx =
        var_db->registerVariableAndContext(d_div_var, var_db->getContext(d_object_name + "CHI::SCRATCH"), no_ghosts);

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
BrinkmanPenalizationAdvDiff::preprocessBrinkmanPenalizationAdvDiff(double current_time,
                                                                   double /*new_time*/,
                                                                   int /*num_cycles*/)
{
    Pointer<PatchHierarchy<NDIM> > patch_hierarchy = d_adv_diff_solver->getPatchHierarchy();
    const int coarsest_ln = 0;
    const int finest_ln = patch_hierarchy->getFinestLevelNumber();

    // Allocate required patch data
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM> > level = patch_hierarchy->getPatchLevel(ln);
        level->allocatePatchData(d_B_scratch_idx, current_time);
        level->allocatePatchData(d_B_chi_scratch_idx, current_time);
        level->allocatePatchData(d_div_B_scratch_idx, current_time);
        level->allocatePatchData(d_div_B_chi_scratch_idx, current_time);
        level->allocatePatchData(d_chi_scratch_idx, current_time);
    }

    return;
} // preprocessBrinkmanPenalizationAdvDiff

void
BrinkmanPenalizationAdvDiff::postprocessBrinkmanPenalizationAdvDiff(double /*current_time*/,
                                                                    double /*new_time*/,
                                                                    int /*num_cycles*/)
{
    Pointer<PatchHierarchy<NDIM> > patch_hierarchy = d_adv_diff_solver->getPatchHierarchy();
    const int coarsest_ln = 0;
    const int finest_ln = patch_hierarchy->getFinestLevelNumber();

    // Deallocate patch data
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM> > level = patch_hierarchy->getPatchLevel(ln);
        level->deallocatePatchData(d_B_scratch_idx);
        level->deallocatePatchData(d_B_chi_scratch_idx);
        level->deallocatePatchData(d_div_B_scratch_idx);
        level->deallocatePatchData(d_div_B_chi_scratch_idx);
        level->deallocatePatchData(d_chi_scratch_idx);
    }
    return;
} // postprocessBrinkmanPenalizationAdvDiff

void
BrinkmanPenalizationAdvDiff::setPenaltyCoefficient(double eta)
{
    // TODO: Allow for flexibility in setting eta for different Q and level sets?
    d_eta = eta;
    return;
} // setPenaltyCoefficient

void
BrinkmanPenalizationAdvDiff::registerDirichletHomogeneousNeumannBC(Pointer<CellVariable<NDIM, double> > Q_var,
                                                                   Pointer<CellVariable<NDIM, double> > ls_solid_var,
                                                                   std::string bc_type,
                                                                   double bc_val)
{
#if !defined(NDEBUG)
    TBOX_ASSERT(Q_var);
    TBOX_ASSERT(ls_solid_var);
#endif
    auto bc_type_enum = string_to_enum<AdvDiffBrinkmanPenalizationBcType>(bc_type);
    if (bc_type_enum != DIRICHLET && bc_type_enum != NEUMANN)
    {
        TBOX_ERROR("BrinkmanPenalizationAdvDiff::registerDirichletHomogeneousNeumannBC\n"
                   << "  unsupported Brinkman penalization BC type: "
                   << enum_to_string<AdvDiffBrinkmanPenalizationBcType>(bc_type_enum) << " \n"
                   << "  valid choices are: DIRICHLET, NEUMANN\n");
    }

    if (bc_type_enum == NEUMANN && bc_val != 0.0)
    {
        TBOX_ERROR("BrinkmanPenalizationAdvDiff::registerDirichletHomogeneousNeumannBC\n"
                   << "  this function can only be used to register homogeneous/inhomogeneous Dirichlet BCs\n"
                   << "  and homogeneous Neumann BCs. Inhomogeneous Neumann BCs must be registered with\n"
                   << "  BrinkmanPenalizationAdvDiff::registerDirichletHomogeneousNeumannBC\n");
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
                    TBOX_ERROR("BrinkmanPenalizationAdvDiff::registerDirichletHomogeneousNeumannBC\n"
                               << "  two separate boundary conditions on level set variable " << ls_solid_var->getName()
                               << " \n"
                               << "  are being registered for transported quantity " << Q_var->getName() << "\n");
                }
            }
        }
    }

    // No user supplied callbacks are needed
    BrinkmanInhomogeneousNeumannBCsFcnPtr callback = nullptr;
    void* ctx = nullptr;
    auto bc_tuple = std::make_tuple(ls_solid_var, bc_type_enum, bc_val, callback, ctx);
    d_Q_bc[Q_var].push_back(bc_tuple);
    return;
} // registerDirichletHomogeneousNeumannBC

void
BrinkmanPenalizationAdvDiff::registerDirichletHomogeneousNeumannBC(
    Pointer<CellVariable<NDIM, double> > Q_var,
    std::vector<Pointer<CellVariable<NDIM, double> > > ls_solid_vars,
    std::vector<string> bc_types,
    std::vector<double> bc_vals)
{
    TBOX_ASSERT(ls_solid_vars.size() == bc_types.size() == bc_vals.size());
    for (int i = 0; i < ls_solid_vars.size(); ++i)
    {
        registerDirichletHomogeneousNeumannBC(Q_var, ls_solid_vars[i], bc_types[i], bc_vals[i]);
    }
    return;
} // registerDirichletHomogeneousNeumannBC

void
BrinkmanPenalizationAdvDiff::registerInhomogeneousNeumannBC(Pointer<CellVariable<NDIM, double> > Q_var,
                                                            Pointer<CellVariable<NDIM, double> > ls_solid_var,
                                                            BrinkmanInhomogeneousNeumannBCsFcnPtr callback,
                                                            void* ctx)
{
#if !defined(NDEBUG)
    TBOX_ASSERT(Q_var);
    TBOX_ASSERT(ls_solid_var);
    TBOX_ASSERT(callback);
    TBOX_ASSERT(ctx);
#endif
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
                    TBOX_ERROR("BrinkmanPenalizationAdvDiff::registerInhomogeneousNeumannBC\n"
                               << "  two separate boundary conditions on level set variable " << ls_solid_var->getName()
                               << " \n"
                               << "  are being registered for transported quantity " << Q_var->getName() << "\n");
                }
            }
        }
    }

    // Specifying a dummy nonzero value for bc_val since this input value will not be used for inhomogeneous Neumann BCs
    auto bc_type_enum = string_to_enum<AdvDiffBrinkmanPenalizationBcType>("NEUMANN");
    double bc_val = std::numeric_limits<double>::max();
    auto bc_tuple = std::make_tuple(ls_solid_var, bc_type_enum, bc_val, callback, ctx);
    d_Q_bc[Q_var].push_back(bc_tuple);
    return;
} // registerInhomogeneousNeumannBC

void
BrinkmanPenalizationAdvDiff::computeDampingCoefficient(int C_idx,
                                                       Pointer<CellVariable<NDIM, double> > Q_var,
                                                       double lambda)
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

        // Get the solid level set info
        VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
        const int phi_idx = var_db->mapVariableAndContextToIndex(ls_solid_var, d_adv_diff_solver->getNewContext());

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
                    const double chi = 1.0 - Hphi;
                    if (bc_type == DIRICHLET)
                    {
                        (*C_data)(ci) = lambda + chi * 1.0 / d_eta;
                    }
                    else if (bc_type == NEUMANN)
                    {
                        (*C_data)(ci) = (1.0 - chi) * lambda;
                    }
                    else
                    {
                        // This statement should not be reached
                        TBOX_ERROR("Error in BrinkmanPenalizationAdvDiff::computeDampingCoefficient: \n"
                                   << "bc_type = " << enum_to_string(bc_type) << "\n");
                    }
                }
            }
        }
    }
    return;
} // computeDampingCoefficient

void
BrinkmanPenalizationAdvDiff::computeDiffusionCoefficient(int D_idx,
                                                         Pointer<CellVariable<NDIM, double> > Q_var,
                                                         int kappa_idx,
                                                         double kappa)
{
#if !defined(NDEBUG)
    TBOX_ASSERT(d_Q_bc.find(Q_var) != d_Q_bc.end());
#endif
    auto brinkman_zones = d_Q_bc[Q_var];
    const bool variable_kappa = (kappa_idx != -1);
    for (const auto& bc_tup : brinkman_zones)
    {
        // Get the BC specifications for each zone
        Pointer<CellVariable<NDIM, double> > ls_solid_var = std::get<0>(bc_tup);
        AdvDiffBrinkmanPenalizationBcType bc_type = std::get<1>(bc_tup);

        // Get the solid level set info
        VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
        const int phi_idx = var_db->mapVariableAndContextToIndex(ls_solid_var, d_adv_diff_solver->getNewContext());
        const int phi_scratch_idx =
            var_db->mapVariableAndContextToIndex(ls_solid_var, d_adv_diff_solver->getScratchContext());

        Pointer<PatchHierarchy<NDIM> > patch_hierarchy = d_adv_diff_solver->getPatchHierarchy();
        const int coarsest_ln = 0;
        const int finest_ln = patch_hierarchy->getFinestLevelNumber();

        // Fill the ghost cells of phi_scratch
        typedef HierarchyGhostCellInterpolation::InterpolationTransactionComponent InterpolationTransactionComponent;
        std::vector<InterpolationTransactionComponent> phi_transaction_comps(1);
        phi_transaction_comps[0] =
            InterpolationTransactionComponent(phi_scratch_idx,
                                              phi_idx,
                                              "CONSERVATIVE_LINEAR_REFINE",
                                              false,
                                              "CONSERVATIVE_COARSEN",
                                              "LINEAR",
                                              false,
                                              d_adv_diff_solver->getPhysicalBcCoefs(ls_solid_var));
        Pointer<HierarchyGhostCellInterpolation> hier_bdry_fill = new HierarchyGhostCellInterpolation();
        hier_bdry_fill->initializeOperatorState(phi_transaction_comps, patch_hierarchy);
        hier_bdry_fill->fillData(d_current_time);

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

                Pointer<CellData<NDIM, double> > ls_solid_data = patch->getPatchData(phi_scratch_idx);
                Pointer<SideData<NDIM, double> > D_data = patch->getPatchData(D_idx);

                // There is no Brinkman contribution to the diffusion coefficient for Dirichlet BCs
                if (bc_type == DIRICHLET)
                {
                    if (!variable_kappa)
                    {
                        D_data->fillAll(kappa);
                    }
                    else
                    {
                        Pointer<SideData<NDIM, double> > kappa_data = patch->getPatchData(kappa_idx);
                        D_data->copy(*kappa_data);
                    }
                }
                else if (bc_type == NEUMANN)
                {
                    Pointer<SideData<NDIM, double> > kappa_data =
                        variable_kappa ? patch->getPatchData(kappa_idx) : nullptr;
                    for (unsigned int axis = 0; axis < NDIM; ++axis)
                    {
                        for (Box<NDIM>::Iterator it(SideGeometry<NDIM>::toSideBox(patch_box, axis)); it; it++)
                        {
                            SideIndex<NDIM> s_i(it(), axis, SideIndex<NDIM>::Lower);

                            const double phi_lower = (*ls_solid_data)(s_i.toCell(0));
                            const double phi_upper = (*ls_solid_data)(s_i.toCell(1));
                            const double phi = 0.5 * (phi_lower + phi_upper);
                            const double kp = variable_kappa ? (*kappa_data)(s_i) : kappa;
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

                            const double chi = 1.0 - Hphi;
                            (*D_data)(s_i) = (1.0 - chi) * kp + d_eta * chi;
                        }
                    }
                }
                else
                {
                    // This statement should not be reached
                    TBOX_ERROR("Error in BrinkmanPenalizationAdvDiff::computeDiffusionCoefficient: \n"
                               << "bc_type = " << enum_to_string(bc_type) << "\n");
                }
            }
        }
    }
    return;
} // computeDiffusionCoefficient

void
BrinkmanPenalizationAdvDiff::computeForcing(int F_idx, Pointer<CellVariable<NDIM, double> > Q_var)
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

        // Dirichlet and homogeneous Neumann case, which do not require callback functions
        bool requires_callbacks = (bc_type == NEUMANN && bc_val != 0.0);

        Pointer<PatchHierarchy<NDIM> > patch_hierarchy = d_adv_diff_solver->getPatchHierarchy();
        const int coarsest_ln = 0;
        const int finest_ln = patch_hierarchy->getFinestLevelNumber();

        if (!requires_callbacks)
        {
            // Get the solid level set info
            VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
            const int phi_idx = var_db->mapVariableAndContextToIndex(ls_solid_var, d_adv_diff_solver->getNewContext());

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
                    Pointer<CellData<NDIM, double> > F_data = patch->getPatchData(F_idx);

                    if (bc_type == DIRICHLET)
                    {
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
                            const double chi = (1.0 - Hphi);
                            (*F_data)(ci) = chi / d_eta * bc_val;
                        }
                    }
                    else if (bc_type == NEUMANN)
                    {
#if !defined(NDEBUG)
                        TBOX_ASSERT(bc_val == 0.0);
#endif
                        // For homogeneous Neumann conditions, there is no additional Brinkman forcing
                        F_data->fillAll(0.0);
                    }
                    else
                    {
                        // This statement should not be reached
                        TBOX_ERROR("Error in BrinkmanPenalizationAdvDiff::computeForcing: \n"
                                   << "bc_type = " << enum_to_string(bc_type) << "\n");
                    }
                }
            }
        }
        else
        {
            // Fill in patch data from user-prescribed callback
            BrinkmanInhomogeneousNeumannBCsFcnPtr bp_fcn = std::get<3>(bc_tup);
            void* bp_ctx = std::get<4>(bc_tup);
            Pointer<HierarchyMathOps> hier_math_ops = d_adv_diff_solver->getHierarchyMathOps();
            bp_fcn(d_B_scratch_idx, hier_math_ops, d_current_time, bp_ctx);

            // Compute chi*B throughout the hierarchy
            VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
            const int phi_idx = var_db->mapVariableAndContextToIndex(ls_solid_var, d_adv_diff_solver->getNewContext());
            const int phi_scratch_idx =
                var_db->mapVariableAndContextToIndex(ls_solid_var, d_adv_diff_solver->getScratchContext());

            // Fill the ghost cells of phi_scratch
            typedef HierarchyGhostCellInterpolation::InterpolationTransactionComponent
                InterpolationTransactionComponent;
            std::vector<InterpolationTransactionComponent> phi_transaction_comps(1);
            phi_transaction_comps[0] =
                InterpolationTransactionComponent(phi_scratch_idx,
                                                  phi_idx,
                                                  "CONSERVATIVE_LINEAR_REFINE",
                                                  false,
                                                  "CONSERVATIVE_COARSEN",
                                                  "LINEAR",
                                                  false,
                                                  d_adv_diff_solver->getPhysicalBcCoefs(ls_solid_var));
            Pointer<HierarchyGhostCellInterpolation> hier_bdry_fill = new HierarchyGhostCellInterpolation();
            hier_bdry_fill->initializeOperatorState(phi_transaction_comps, patch_hierarchy);
            hier_bdry_fill->fillData(d_current_time);
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

                    Pointer<CellData<NDIM, double> > ls_solid_data = patch->getPatchData(phi_scratch_idx);
                    Pointer<SideData<NDIM, double> > B_data = patch->getPatchData(d_B_scratch_idx);
                    Pointer<SideData<NDIM, double> > B_chi_data = patch->getPatchData(d_B_chi_scratch_idx);
                    Pointer<CellData<NDIM, double> > chi_data = patch->getPatchData(d_chi_scratch_idx);

                    for (unsigned int axis = 0; axis < NDIM; ++axis)
                    {
                        for (Box<NDIM>::Iterator it(SideGeometry<NDIM>::toSideBox(patch_box, axis)); it; it++)
                        {
                            SideIndex<NDIM> s_i(it(), axis, SideIndex<NDIM>::Lower);

                            const double phi_lower = (*ls_solid_data)(s_i.toCell(0));
                            const double phi_upper = (*ls_solid_data)(s_i.toCell(1));
                            const double phi = 0.5 * (phi_lower + phi_upper);
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

                            const double chi = 1.0 - Hphi;
                            (*B_chi_data)(s_i) = chi * (*B_data)(s_i);
                        }
                    }
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
                        (*chi_data)(ci) = (1.0 - Hphi);
                    }
                }
            }

            // Compute F = div(chi * B) - chi * div(B) throughout the hierarchy.
            HierarchyCellDataOpsReal<NDIM, double> hier_cc_data_ops(patch_hierarchy, coarsest_ln, finest_ln);
            HierarchySideDataOpsReal<NDIM, double> hier_sc_data_ops(patch_hierarchy, coarsest_ln, finest_ln);
            hier_math_ops->div(
                d_div_B_chi_scratch_idx, d_div_var, 1.0, d_B_chi_scratch_idx, d_B_var, NULL, d_current_time, false);
            hier_math_ops->div(
                d_div_B_scratch_idx, d_div_var, 1.0, d_B_scratch_idx, d_B_var, NULL, d_current_time, false);
            hier_cc_data_ops.multiply(d_div_B_scratch_idx, d_chi_scratch_idx, d_div_B_scratch_idx);
            hier_cc_data_ops.axmy(F_idx, 1.0, d_div_B_chi_scratch_idx, d_div_B_scratch_idx);

            // Nullify patch data just to be safe
            hier_sc_data_ops.setToScalar(d_B_scratch_idx, std::numeric_limits<double>::quiet_NaN());
            hier_sc_data_ops.setToScalar(d_B_chi_scratch_idx, std::numeric_limits<double>::quiet_NaN());
            hier_cc_data_ops.setToScalar(d_div_B_chi_scratch_idx, std::numeric_limits<double>::quiet_NaN());
            hier_cc_data_ops.setToScalar(d_div_B_scratch_idx, std::numeric_limits<double>::quiet_NaN());
            hier_cc_data_ops.setToScalar(d_chi_scratch_idx, std::numeric_limits<double>::quiet_NaN());
        }
    }
    return;
} // computeForcing

void
BrinkmanPenalizationAdvDiff::maskForcingTerm(int N_idx, Pointer<CellVariable<NDIM, double> > Q_var)
{
#if !defined(NDEBUG)
    TBOX_ASSERT(d_Q_bc.find(Q_var) != d_Q_bc.end());
#endif
    auto brinkman_zones = d_Q_bc[Q_var];
    for (const auto& bc_tup : brinkman_zones)
    {
        // Get the BC specifications for each zone
        AdvDiffBrinkmanPenalizationBcType bc_type = std::get<1>(bc_tup);

        if (bc_type == DIRICHLET) continue;
#if !defined(NDEBUG)
        TBOX_ASSERT(bc_type == NEUMANN);
#endif

        // Get the solid level set info
        Pointer<CellVariable<NDIM, double> > ls_solid_var = std::get<0>(bc_tup);
        VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
        const int phi_idx = var_db->mapVariableAndContextToIndex(ls_solid_var, d_adv_diff_solver->getCurrentContext());

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
                Pointer<CellData<NDIM, double> > N_data = patch->getPatchData(N_idx);

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
                    const double chi = (1.0 - Hphi);
                    (*N_data)(ci) = (1.0 - chi) * (*N_data)(ci);
                }
            }
        }
    }
    return;
} // maskForcingTerm

/////////////////////////////// PRIVATE //////////////////////////////////////

/////////////////////////////// NAMESPACE ////////////////////////////////////

} // namespace IBAMR

//////////////////////////////////////////////////////////////////////////////
