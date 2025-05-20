#ifndef CONCRETE_DAMAGE_PLASTICITY_H
#define CONCRETE_DAMAGE_PLASTICITY_H

#include <Eigen/Dense>
#include <map>
#include <string>

class ConcreteDamagePlasticity {
public:
    // 构造函数，初始化材料属性
    ConcreteDamagePlasticity(const std::map<std::string, double>& material_properties);
    
    // 计算弹塑性张量
    Eigen::Matrix3d compute_elasto_plastic_tensor(const Eigen::VectorXd& stress,
                                                  std::map<std::string, double>* state_vars);

private:
    // 材料属性
    double ft_, fc_;              // 拉伸强度，压缩强度
    double alpha_t_, alpha_c_;    // 张拉与压缩的软化速率参数
    double ftr_, fcr_;            // 残余强度
    double epsilon_p_t0_, epsilon_p_c0_;  // 初始塑性应变阈值
    
    // 更新的损伤变量
    double damage_t_, damage_c_; // 张拉与压缩的损伤变量
};

#endif // CONCRETE_DAMAGE_PLASTICITY_H
