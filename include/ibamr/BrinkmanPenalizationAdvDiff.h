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

#ifndef included_BrinkmanPenalizationAdvDiff
#define included_BrinkmanPenalizationAdvDiff

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ibamr/ibamr_enums.h"

#include "ibtk/ibtk_macros.h"
#include "ibtk/ibtk_utilities.h"

#include "tbox/Database.h"
#include "tbox/Pointer.h"

#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace IBAMR
{
class AdvDiffHierarchyIntegrator;
} // namespace IBAMR
namespace SAMRAI
{
namespace pdat
{
template <int DIM, class TYPE>
class CellVariable;
} // namespace pdat
} // namespace SAMRAI

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace IBAMR
{
/*!
 * \brief BrinkmanPenalizationAdvDiff is an abstract class that provides an interface
 * to implement Brinkman penalization body force in the advection-diffusion equation
 * in order to enforce Dirichlet and Neumann boundary conditions on surfaces of rigid
 * immersed bodies. A single instance of this class is meant to handle all of the Brinkman
 * penalization zones for multiple transported quantities with various boundary conditions.

 * TODO: A description of the penalization terms for both Dirichlet and Neumann BCs
 * For Dirichlet BCs, inhomogeneous and homogeneous BCs are treated in the same way.
 * For Neumann BCs, homogeneous BCs are fairly straightforward, while inhomogeneous BCs require
 * some additional input by the user. Add a reference to Sakurai's paper
 */
class BrinkmanPenalizationAdvDiff
{
public:
    /*
     * \brief Constructor of the class.
     */
    BrinkmanPenalizationAdvDiff(std::string object_name,
                                SAMRAI::tbox::Pointer<IBAMR::AdvDiffHierarchyIntegrator> adv_diff_solver);

    /*
     * \brief Destructor of the class.
     */
    ~BrinkmanPenalizationAdvDiff() = default;

    /*!
     * \brief Set the time interval in which Brinkman forcing is computed.
     */
    void setTimeInterval(double current_time, double new_time);

    /*!
     * \brief Preprocess routine before computing Brinkman penalization terms.
     *
     */
    void preprocessBrinkmanPenalizationAdvDiff(double current_time, double new_time, int num_cycles);

    /*!
     * \brief Postprocess routine after computing Brinkman penalization terms.
     *
     */
    void postprocessBrinkmanPenalizationAdvDiff(double current_time, double new_time, int num_cycles);

    /*!
     * \brief Set Brinkman penalization penalty factor.
     */
    void setBrinkmanCoefficient(double eta);

    /*
     * \brief Get the name of the object.
     */
    const std::string& getName() const
    {
        return d_object_name;
    } // getName

    /*
     * \brief Get the Brinkman coefficient.
     */
    double getBrinkmanCoefficient() const
    {
        return d_eta;
    } // getBrinkmanCoefficient

    /*
     * \brief Get the current time interval \f$ [t^{n+1}, t^n] \f$ in which Brinkman
     * velocity is computed.
     */
    std::pair<double, double> getCurrentTimeInterval() const
    {
        return std::make_pair(d_new_time, d_current_time);
    } // getCurrentTimeInterval

    /*
     * \brief Function to determine if a transported quantity has Brinkman boundary conditions
     * associated with is.
     */
    bool hasBrinkmanBoundaryCondition(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double> > Q_var) const
    {
        return d_Q_bc.find(Q_var) != d_Q_bc.end();
    } // hasBrinkmanBoundaryCondition

    /*!
     * \brief Function specifying the optional forcing function for inhomogeneous Neumann boundary conditions \f$ n \dot
     * \Q = \gamma \f$
     *
     * The user must set the patch data B_idx such that \f$ n \dot B = \gamma \f$.
     * The applied forcing term is then computed internally as \f$ \nabla \dot (\chi B) - \chi \nabla \dot B \f$.
     * Note that B_idx contains side-centered patch data.
     */
    using BrinkmanInhomogeneousNeumannBCsFcnPtr = void (*)(int B_idx,
                                                           SAMRAI::tbox::Pointer<IBTK::HierarchyMathOps> hier_math_ops,
                                                           double time,
                                                           void* ctx);

    /*
     * \brief Register a transported quantity with this object, along with the solid level set
     * variable for which to apply a specified boundary condition.
     *
     * \note This function can only be used to register homogeneous/inhomogeneous Dirichlet BCs, and homogeneous Neumann
     * BCs.
     */

    void
    registerBrinkmanBoundaryCondition(SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double> > Q_var,
                                      SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double> > ls_solid_var,
                                      std::string bc_type,
                                      double bc_val);

    /*
     * \brief Register a transported quantity with this object, along with multiple solid level set
     * variables for which to apply a specified boundary condition.
     *
     * \note This function can only be used to register homogeneous/inhomogeneous Dirichlet BCs, and homogeneous Neumann
     * BCs.
     */
    void registerBrinkmanBoundaryCondition(
        SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double> > Q_var,
        std::vector<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double> > > ls_solid_vars,
        std::vector<std::string> bc_types,
        std::vector<double> bc_vals);

    /*
     * \brief Register a transported quantity with this object, along with the solid level set variable
     * on which to apply inhomogeneous Neumann boundary conditions.
     *
     * \note Inhomogeneous Neumann BCs are treated uniquely within this class and require additional user callback
     * inputs.
     */
    void registerInhomogeneousNeumannBrinkmanBoundaryCondition(
        SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double> > Q_var,
        SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double> > ls_solid_var,
        BrinkmanInhomogeneousNeumannBCsFcnPtr callback,
        void* ctx);

    /*
     * \brief Function to compute the cell-centered coefficient to the damping linear operator
     * and RHS of the advection-diffusion equation for a specified transported quantity Q_var with damping
     * coefficient lambda.
     *
     * For Dirichlet BCs, \f$ C = \lambda + \chi/\eta \f$ where \f$\chi = 1-H\f$.
     * For Neumann BCs, \f$ C = (1 - \chi) \lambda + \f$ where \f$\chi = 1-H\f$.
     */
    void computeBrinkmanDampingCoefficient(int C_idx,
                                           SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double> > Q_var,
                                           double lambda);

    /*
     * \brief Function to compute the side-centered coefficient to the diffusion linear operator
     * and RHS of the advection-diffusion equation for a specified transported quantity Q_var with diffusion
     * coefficient kappa. Note that this function is able to handle both constant and variable kappa.
     *
     * For Dirichlet BCs, \f$ D = kappa \f$.
     * For Neumann BCs, \f$ D = (1 - \chi) \kappa + \eta \chi \lambda + \f$ where \f$\chi = 1-H\f$
     */
    void computeBrinkmanDiffusionCoefficient(int D_idx,
                                             SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double> > Q_var,
                                             int kappa_idx,
                                             double kappa);

    /*
     * \brief Function to compute the Brinkman forcing contribution to the RHS of the advection-diffusion solver
     * for a specified transported quantity Q_var.
     *
     * For Dirichlet BCs, \f$ F = \chi/\eta Q_{bc}\f$ where \f$\chi = 1-H\f$.
     * For homogeneous Neumann BCs, \f$ F = 0\f$.
     * For inhomogeneous Neumann BCs, \f$ F = \nabla \dot (\chi B) - \chi \nabla \dot B \f$, with a user defined
     * \f$B\f$.
     */
    void computeBrinkmanForcing(int F_idx, SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double> > Q_var);

    /*
     * \brief Function to mask the additional forcing terms on the RHS of the advection-diffusion solver
     * e.g. \f$ u \dot \grad Q\f$ and body forces.
     *
     * For Dirichlet BCs, no masking function is applied.
     * For Neumann BCs, \f$ N = (1-\chi) N\f$ where \f$\chi = 1-H\f$.
     */
    void maskForcingTerm(int N_idx, SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double> > Q_var);

private:
    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    BrinkmanPenalizationAdvDiff(const BrinkmanPenalizationAdvDiff& from) = delete;

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    BrinkmanPenalizationAdvDiff& operator=(const BrinkmanPenalizationAdvDiff& that) = delete;

    /*
     * Object name.
     */
    std::string d_object_name;

    /*
     * Pointer to the adv-diff solver.
     */
    SAMRAI::tbox::Pointer<IBAMR::AdvDiffHierarchyIntegrator> d_adv_diff_solver;

    /*
     * Time interval
     */
    double d_current_time = std::numeric_limits<double>::quiet_NaN(),
           d_new_time = std::numeric_limits<double>::quiet_NaN();

    /*
     * Brinkman coefficient.
     */
    double d_eta = 1e-4;

    /*
     * Map for storing transported variables and various registered BCs
     */
    std::map<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double> >,
             std::vector<std::tuple<SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double> >,
                                    AdvDiffBrinkmanPenalizationBcType,
                                    double,
                                    BrinkmanInhomogeneousNeumannBCsFcnPtr,
                                    void*> > >
        d_Q_bc;

    /*
     * Patch data required for computing additional forcing for inhomogeneous Neumann BCs
     */
    SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double> > d_B_var;
    int d_B_scratch_idx, d_B_chi_scratch_idx;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double> > d_div_var;
    int d_div_B_scratch_idx, d_div_B_chi_scratch_idx, d_chi_scratch_idx;
};

} // namespace IBAMR

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_BrinkmanPenalizationAdvDiff
