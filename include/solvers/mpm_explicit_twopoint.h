#ifndef MPM_MPM_EXPLICIT_TWOPOINT_H_
#define MPM_MPM_EXPLICIT_TWOPOINT_H_

#ifdef USE_GRAPH_PARTITIONING

#include "graph.h"
#endif

#include "mpm_base.h"

namespace mpm {

//! MPMExplicitTwoPoint class
//! \brief A class that implements the fully explicit one phase mpm
//! \details A single-phase explicit MPM
//! \tparam Tdim Dimension
template <unsigned Tdim>
class MPMExplicitTwoPoint : public MPMBase<Tdim> {
 public:
  //! Default constructor
  MPMExplicitTwoPoint(const std::shared_ptr<IO>& io);

  //! Solve
  bool solve() override;

 protected:
  // Generate a unique id for the analysis
  using mpm::MPMBase<Tdim>::uuid_;
  //! Time step size
  using mpm::MPMBase<Tdim>::dt_;
  //! Current step
  using mpm::MPMBase<Tdim>::step_;
  //! Number of steps
  using mpm::MPMBase<Tdim>::nsteps_;
  //! Number of steps
  using mpm::MPMBase<Tdim>::nload_balance_steps_;
  //! Output steps
  using mpm::MPMBase<Tdim>::output_steps_;
  //! A unique ptr to IO object
  using mpm::MPMBase<Tdim>::io_;
  //! JSON analysis object
  using mpm::MPMBase<Tdim>::analysis_;
  //! JSON post-process object
  using mpm::MPMBase<Tdim>::post_process_;
  //! Logger
  using mpm::MPMBase<Tdim>::console_;
  //! MPM Scheme
  using mpm::MPMBase<Tdim>::mpm_scheme_;
  //! Stress update method
  using mpm::MPMBase<Tdim>::stress_update_;
  //! Interface scheme
  using mpm::MPMBase<Tdim>::contact_;

#ifdef USE_GRAPH_PARTITIONING
  //! Graph
  using mpm::MPMBase<Tdim>::graph_;
#endif

  //! velocity update
  using mpm::MPMBase<Tdim>::velocity_update_;
  //! FLIP-PIC blending ratio
  using mpm::MPMBase<Tdim>::blending_ratio_;
  //! Gravity
  using mpm::MPMBase<Tdim>::gravity_;
  //! Mesh object (solid phase)
  using mpm::MPMBase<Tdim>::mesh_;
  //! Materials
  using mpm::MPMBase<Tdim>::materials_;
  //! Node concentrated force
  using mpm::MPMBase<Tdim>::set_node_concentrated_force_;
  //! Damping type
  using mpm::MPMBase<Tdim>::damping_type_;
  //! Damping factor
  using mpm::MPMBase<Tdim>::damping_factor_;
  //! Locate particles
  using mpm::MPMBase<Tdim>::locate_particles_;
  //! Constraints Pointer
  using mpm::MPMBase<Tdim>::constraints_;
  //! Absorbing Boundary
  using mpm::MPMBase<Tdim>::absorbing_boundary_;

  /**
   * \defgroup Interface Variables (includes multimaterial and levelset)
   * @{
   */
  //! Interface boolean
  using mpm::MPMBase<Tdim>::interface_;
  //! Interface type
  using mpm::MPMBase<Tdim>::interface_type_;
  //! Levelset damping factor
  using mpm::MPMBase<Tdim>::levelset_damping_;
  //! Levelset PIC contact velocity
  using mpm::MPMBase<Tdim>::levelset_pic_;
  //! Levelset violation correction factor
  using mpm::MPMBase<Tdim>::levelset_violation_corrector_;
  /**@}*/

  //! Fluid mesh pointer for coupled fluid phase
  std::shared_ptr<mpm::Mesh<Tdim>> mesh_fluid_;

  //! Initialise mesh (override) to handle solid and fluid meshes
  void initialise_mesh() override;

 private:
  //! Pressure smoothing
  bool pressure_smoothing_{false};
  std::shared_ptr<Mesh<Tdim>> fluid_mesh_;
};  // MPMExplicitTwoPoint class

}  // namespace mpm

#include "mpm_explicit_twopoint.tcc"

#endif  // MPM_MPM_EXPLICIT_TWOPOINT_H_
