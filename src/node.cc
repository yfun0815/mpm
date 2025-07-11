#include "node.h"
#include "factory.h"
#include "node_base.h"
#include "node_levelset.h"

// Node2D (2 DoF, 1 Phase)
static Register<mpm::NodeBase<2>, mpm::Node<2, 2, 1>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    node2d("N2D");

// Node3D (3 DoF, 1 Phase)
static Register<mpm::NodeBase<3>, mpm::Node<3, 3, 1>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    node3d("N3D");

// Node2D (2 DoF, 2 Phase)
static Register<mpm::NodeBase<2>, mpm::Node<2, 2, 2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    node2d2phase("N2D2P");

// Node3D (3 DoF, 2 Phase)
static Register<mpm::NodeBase<3>, mpm::Node<3, 3, 2>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    node3d2phase("N3D2P");

// Node2D (2 DoF, 1 Phase)
static Register<mpm::NodeBase<2>, mpm::NodeLevelset<2, 2, 1>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    node2dlevelset("N2DLS");

// Node3D (3 DoF, 1 Phase)
static Register<mpm::NodeBase<3>, mpm::NodeLevelset<3, 3, 1>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    node3dlevelset("N3DLS");