#ifndef MPM_MPM_BASE_H_
#define MPM_MPM_BASE_H_

#include <numeric>
#include <sstream>

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

// CSV-parser
#include "csv/csv.h"

// MPI
#ifdef USE_MPI
#include "mpi.h"
#endif

#ifdef USE_GRAPH_PARTITIONING
#include "graph.h"
#endif

#include "constraints.h"
#include "contact.h"
#include "contact_friction.h"
#include "contact_levelset.h"
#include "mpm.h"
#include "mpm_scheme.h"
#include "mpm_scheme_musl.h"
#include "mpm_scheme_newmark.h"
#include "mpm_scheme_usf.h"
#include "mpm_scheme_usl.h"
#include "particle.h"
#include "solver_base.h"
#include "vector.h"

namespace mpm {

//! Variable type
//! Scalar: boolean, unsigned, int, double
//! Vector: Vector of size 3
//! Tensor: Symmetric tensor arranged in voigt notation
enum class VariableType { Scalar, Vector, Tensor };

//! Damping type
//! None: No damping is specified
//! Cundall: Cundall damping
enum class Damping { None, Cundall };

//! Velocity update type
extern std::map<std::string, mpm::VelocityUpdate> VelocityUpdateType;

//! MPMBase class
//! \brief A class that implements the fully base one phase mpm
//! \details A Base MPM class
//! \tparam Tdim Dimension
template <unsigned Tdim>
class MPMBase : public MPM {
 public:
  //! Default constructor
  MPMBase(const std::shared_ptr<IO>& io);

  //! Initialise mesh
  virtual void initialise_mesh();

  //! Initialise particles
  void initialise_particles() override;

  //! Initialise materials
  void initialise_materials() override;

  //! Initialise loading
  void initialise_loads() override;

  //! Initialise math functions
  bool initialise_math_functions(const Json&) override;

  //! Solve
  bool solve() override { return true; }

  //! Checkpoint resume
  bool checkpoint_resume() override;

  //! Domain decomposition
  //! \param[in] initial_step Start of simulation or later steps
  void mpi_domain_decompose(bool initial_step = false) override;

  //! Output results
  //! \param[in] step Time step
  void write_outputs(mpm::Index step) override;

  //! Pressure smoothing
  //! \param[in] phase Phase to smooth pressure
  void pressure_smoothing(unsigned phase);

  //! Particle entity sets
  //! \param[in] check Check duplicates
  void particle_entity_sets(bool check);

  //! Particle velocity constraints
  void particle_velocity_constraints();

  //! Apply Absorbing Constraints
  void nodal_absorbing_constraints();
  //! Delete_particles function
  void delete_particles();

  //! reset_particles_displacement function
  void reset_particles_displacement();

 protected:
  //! Initialise implicit solver
  //! \param[in] lin_solver_props Linear solver properties
  //! \param[in, out] linear_solver Linear solver map
  void initialise_linear_solver(
      const Json& lin_solver_props,
      tsl::robin_map<
          std::string,
          std::shared_ptr<mpm::SolverBase<Eigen::SparseMatrix<double>>>>&
          linear_solver);

  //! Write HDF5 files
  void write_hdf5(mpm::Index step, mpm::Index max_steps) override;

#ifdef USE_VTK
  //! Write VTK files
  void write_vtk(mpm::Index step, mpm::Index max_steps) override;
#endif

#ifdef USE_PARTIO
  //! Write PARTIO files
  void write_partio(mpm::Index step, mpm::Index max_steps) override;
#endif

 private:
  //! Return if a mesh will be isoparametric or not
  //! \retval isoparametric Status of mesh type
  bool is_isoparametric();

  //! Node entity sets
  //! \param[in] mesh_prop Mesh properties
  //! \param[in] check Check duplicates
  void node_entity_sets(const Json& mesh_prop, bool check);

  //! Node Euler angles
  //! \param[in] mesh_prop Mesh properties
  //! \param[in] mesh_io Mesh IO handle
  void node_euler_angles(const Json& mesh_prop,
                         const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io);

  //! Nodal velocity constraints
  //! \param[in] mesh_prop Mesh properties
  //! \param[in] mesh_io Mesh IO handle
  void nodal_velocity_constraints(
      const Json& mesh_prop, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io);

  //! Nodal acceleration constraints
  //! \param[in] mesh_prop Mesh properties
  //! \param[in] mesh_io Mesh IO handle
  void nodal_acceleration_constraints(
      const Json& mesh_prop, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io);

  //! Nodal frictional constraints
  //! \param[in] mesh_prop Mesh properties
  //! \param[in] mesh_io Mesh IO handle
  void nodal_frictional_constraints(
      const Json& mesh_prop, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io);

  //! Nodal adhesional constraints
  //! \param[in] mesh_prop Mesh properties
  //! \param[in] mesh_io Mesh IO handle
  void nodal_adhesional_constraints(
      const Json& mesh_prop, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io);

  //! Nodal pressure constraints
  //! \param[in] mesh_prop Mesh properties
  //! \param[in] mesh_io Mesh IO handle
  void nodal_pressure_constraints(
      const Json& mesh_prop, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io);

  //! Nodal absorbing constraints
  //! \param[in] mesh_prop Mesh properties
  //! \param[in] mesh_io Mesh IO handle
  void nodal_absorbing_constraints(
      const Json& mesh_prop, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io);

  //! Cell entity sets
  //! \param[in] mesh_prop Mesh properties
  //! \param[in] check Check duplicates
  void cell_entity_sets(const Json& mesh_prop, bool check);

  //! Particles cells
  //! \param[in] mesh_prop Mesh properties
  //! \param[in] particle_io Particle IO handle
  void particles_cells(const Json& mesh_prop,
                       const std::shared_ptr<mpm::IOMesh<Tdim>>& particle_io);

  //! Particles volumes
  //! \param[in] mesh_prop Mesh properties
  //! \param[in] particle_io Particle IO handle
  void particles_volumes(const Json& mesh_prop,
                         const std::shared_ptr<mpm::IOMesh<Tdim>>& particle_io);

  //! Particles stresses
  //! \param[in] mesh_prop Mesh properties
  //! \param[in] particle_io Particle IO handle
  void particles_stresses(
      const Json& mesh_prop,
      const std::shared_ptr<mpm::IOMesh<Tdim>>& particle_io);

  // Particles pore pressures
  //! \param[in] mesh_prop Mesh properties
  //! \param[in] particle_io Particle IO handle
  void particles_pore_pressures(
      const Json& mesh_prop,
      const std::shared_ptr<mpm::IOMesh<Tdim>>& particle_io);

  //! Initialise damping
  //! \param[in] damping_props Damping properties
  bool initialise_damping(const Json& damping_props);

  //! Initialise nonlocal mesh
  //! \param[in] mesh_prop Mesh properties
  void initialise_nonlocal_mesh(const Json& mesh_prop);

  //! Initialise particle types
  void initialise_particle_types();

  /**
   * \defgroup Interface Functions (includes multimaterial and levelset)
   */
  /**@{*/
  //! \ingroup Interface
  //! Return if interface and levelset are active
  //! \retval levelset status of mesh
  bool is_levelset();

  //! \ingroup Interface
  //! Nodal levelset inputs
  //! \param[in] mesh_prop Mesh properties
  //! \param[in] mesh_io Mesh IO handle
  void interface_inputs(const Json& mesh_prop,
                        const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io);

  /**@}*/

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Nodal displacement constraints for implicit solver
  //! \ingroup Implicit
  //! \param[in] mesh_prop Mesh properties
  //! \param[in] mesh_io Mesh IO handle
  void nodal_displacement_constraints(
      const Json& mesh_prop, const std::shared_ptr<mpm::IOMesh<Tdim>>& mesh_io);
  /**@}*/

 protected:
  // Generate a unique id for the analysis
  using mpm::MPM::uuid_;
  //! Time step size
  using mpm::MPM::dt_;
  //! Current step
  using mpm::MPM::step_;
  //! Number of steps
  using mpm::MPM::nsteps_;
  //! Output steps
  using mpm::MPM::output_steps_;
  //! Load balancing steps
  using mpm::MPM::nload_balance_steps_;
  //! A unique ptr to IO object
  using mpm::MPM::io_;
  //! JSON analysis object
  using mpm::MPM::analysis_;
  //! JSON post-process object
  using mpm::MPM::post_process_;
  //! Logger
  using mpm::MPM::console_;

  //! Stress update method
  std::string stress_update_{"usf"};
  //! Stress update scheme
  std::shared_ptr<mpm::MPMScheme<Tdim>> mpm_scheme_{nullptr};
  //! Velocity update method
  mpm::VelocityUpdate velocity_update_{mpm::VelocityUpdate::FLIP};
  //! FLIP-PIC blending ratio
  double blending_ratio_{1.0};
  //! Gravity
  Eigen::Matrix<double, Tdim, 1> gravity_;
  //! Mesh object
  std::shared_ptr<mpm::Mesh<Tdim>> mesh_;
  //! Constraints object
  std::shared_ptr<mpm::Constraints<Tdim>> constraints_;
  //! Particle types
  std::set<std::string> particle_types_;
  //! Materials
  std::map<unsigned, std::shared_ptr<mpm::Material<Tdim>>> materials_;
  //! Mathematical functions
  std::map<unsigned, std::shared_ptr<mpm::FunctionBase>> math_functions_;
  //! VTK geometry output bool
  bool geometry_vtk_{false};
  //! VTK particle variables
  tsl::robin_map<mpm::VariableType, std::vector<std::string>> vtk_vars_;
  //! VTK state variables
  tsl::robin_map<unsigned, std::vector<std::string>> vtk_statevars_;
  //! Set node concentrated force
  bool set_node_concentrated_force_{false};
  //! Damping type
  mpm::Damping damping_type_{mpm::Damping::None};
  //! Damping factor
  double damping_factor_{0.};
  //! Locate particles
  bool locate_particles_{true};
  //! Absorbing Boundary Variables
  bool absorbing_boundary_{false};

  /**
   * \defgroup Interface Variables (includes multimaterial and levelset)
   * @{
   */
  //! Interface scheme
  std::shared_ptr<mpm::Contact<Tdim>> contact_{nullptr};
  //! Interface bool
  bool interface_{false};
  //! Interface type
  std::string interface_type_{"none"};
  //! Levelset damping factor
  double levelset_damping_{0.05};
  //! Levelset PIC contact velocity
  bool levelset_pic_{true};
  //! Levelset violation correction factor
  double levelset_violation_corrector_{0.01};
  /**@}*/

  /**
   * \defgroup Nonlocal Variables for nonlocal MPM
   * @{
   */
  // Cell neighbourhood: default 0 for linear element
  unsigned cell_neighbourhood_{0};
  // Node neighbourhood: default 1 for linear element
  unsigned node_neighbourhood_{1};
  /**@}*/

#ifdef USE_GRAPH_PARTITIONING
  // graph pass the address of the container of cell
  std::shared_ptr<Graph<Tdim>> graph_{nullptr};
#endif
};  // MPMBase class
}  // namespace mpm

#include "mpm_base.tcc"

#endif  // MPM_MPM_BASE_H_
