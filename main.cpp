#include <iostream>
#include <vector>
#include <cmath>
#include <locale>
#include <ranges>
#include <random>
#include <limits>
#include <ctime>
#include "Matrix.h"

const static int h_size = 240; // 240 �������� � ������� ����
const static int o_size = 4;  // 4 ������� � �������� ����
const static int i_size = 6; // 6 �������� �� ������� ����

struct Normalization_Params
{
    std::vector<double> mins; 
    std::vector<double> maxs; 
};

// ������� ��� ������������� ������� ����� ����������� ���������
void init_vector(std::vector<double>& vec, double value, size_t size)
{
    vec.resize(size);
    for (size_t i = 0; i < size; ++i)
    {
        vec[i] = value;
    }
}

// ������� ������������ ������ � ��������� [0, 1]
Normalization_Params normalize_to_zero_one(std::vector<std::vector<double>>& data)
{
    Normalization_Params params;
    if (data.empty()) return params;

    size_t vec_size = data[0].size();
    
    init_vector(params.mins, std::numeric_limits<double>::max(), vec_size);
    init_vector(params.maxs, std::numeric_limits<double>::lowest(), vec_size);

    for (const auto& vec : data)
    {
        for (size_t i = 0; i < vec_size; ++i)
        {
            if (vec[i] < params.mins[i])
            {
                params.mins[i] = vec[i];
            }
            if (vec[i] > params.maxs[i])
            {
                params.maxs[i] = vec[i];
            }
        }
    }

    // ����������� ������ � ��������� [0, 1]
    for (auto& vec : data)
    {
        for (size_t i = 0; i < vec_size; ++i)
        {
            if (params.maxs[i] != params.mins[i])
            {
                vec[i] = (vec[i] - params.mins[i]) / (params.maxs[i] - params.mins[i]);
            }
            else
            {
                vec[i] = 0.0; // ���� min = max, �� ������ 0
            }
        }
    }

    return params;
}

// ������� ��� ������������ ������ ������ ������� � ��������� [0, 1]
std::vector<double> normalize_vector(const std::vector<double>& vec, const Normalization_Params& params)
{
    std::vector<double> normalized(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
    {
        if (params.maxs[i] != params.mins[i])
        {
            normalized[i] = (vec[i] - params.mins[i]) / (params.maxs[i] - params.mins[i]);
        }
        else
        {
            normalized[i] = 0.0; // ���� min = max, ������ 0
        }
    }
    return normalized;
}

// ������� ��� �������������� ������� �� [0, 1] � �������� ��������
std::vector<double> denormalize_vector(const std::vector<double>& vec, const Normalization_Params& params)
{
    std::vector<double> denormalized(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
    {
        denormalized[i] = vec[i] * (params.maxs[i] - params.mins[i]) + params.mins[i];
    }
    return denormalized;
}

// ������� ��������� - Leaky ReLU f(x) = x, ���� x > 0, ����� 0.1*x
double leaky_relu(double x)
{
    return x > 0 ? x : 0.1 * x;
}

// ����������� Leaky ReLU
double deriv_leaky_relu(double x)
{
    return x > 0 ? 1.0 : 0.1;
}

// ������� ������ ������� ������� ������������ ������ (mean squared error, MSE):
// n � ��� ���������� �������� �������� (� ������ ������ 4 �� ������ ������)
// y_true � �������� �������� ���������� ("���������� �����") �� ���������� ������ ������
// y_pred � ������������� �������� ����������. ��� ��, ��� ����� ��� ��������� ����
// (y_true - y_pred)^2 ���������� ������������ �������
double mse_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred)
{
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i)
    {
        sum += std::pow(y_true[i] - y_pred[i], 2);
    }
    return sum / y_true.size();
}

class My_First_Neuron_Network
{
    // ������� ��:
    // 6 ������� �������� theta_i, y_i, x_i, t_i, t_i+1, v_i
    // 240 �������� �������� ����
    // 4 �������� �������� y_i+1, x_i+1, theta_i+1, v_i+1
private:
    Matrix<double, h_size, i_size> w;
    /*{
        {w00, w01, w02, w03, w04, w05}, // ��� ������� h[0]
        {w10, w11, w12, w13, w14, w15}, // ��� ������� h[1]
        ...
        {w2390, w2391, w2392, w2393, w2394, w2395} // ��� ������� h[239]
    };*/

    Matrix<double, o_size, h_size> wo;
    /*{
        {wo00, wo01, wo02, ..., wo0239}, // ��� ������� y_pred[0]
        {wo10, wo11, wo12, ..., wo1239}, // ��� ������� y_pred[1]
        {wo20, wo21, wo22, ..., wo2239}, // ��� ������� y_pred[2]
        {wo30, wo31, wo32, ..., wo3239}  // ��� ������� y_pred[3]
    };*/

    std::vector<double> b; // {b0, b1, ..., b239}
    std::vector<double> bo; // {bo0, bo1, bo2, bo3}

public:
    // ������ ��� ��������� ������ ������ ��������, ����� ����� ���� ��������� �� ����������
    // � ����� ������ ������-�� �� ��������� ��� �������� std::vector<double> b(240)
    My_First_Neuron_Network() : b(h_size), bo(o_size)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        // ������������� ����� ������� Xavier ��� �������� ����
        std::normal_distribution<> dis(0.0, std::sqrt(2.0 / (i_size + h_size)));
        for (int i = 0; i < h_size; ++i)
        {
            for (int j = 0; j < i_size; ++j)
            {
                w(i, j) = dis(gen);
            }
        }

        // ������������� ����� ������� Xavier ��� ��������� ����
        dis = std::normal_distribution<>(0.0, std::sqrt(2.0 / (h_size + o_size)));
        for (int i = 0; i < o_size; ++i)
        {
            for (int j = 0; j < h_size; ++j)
            {
                wo(i, j) = dis(gen);
            }
        }

        // ������������� �������� �� �������� ���� � �������� ����
        std::uniform_real_distribution<> bias_dis(-0.1, 0.1);
        for (int i = 0; i < h_size; ++i)
        {
            b[i] = bias_dis(gen);
        }

        // ������������� �������� �� �������� ���� � ��������� ����
        for (int i = 0; i < o_size; ++i)
        {
            bo[i] = bias_dis(gen);
        }
    }

    // ������ ������ ��� ��������� ������������
    std::vector<double> feedforward(const std::vector<double>& input) const
    {
        // ������� �������� �������� �������� ���� 
        std::vector<double> h(h_size); // = { h1, h2, ..., h240 };
        for (int i = 0; i < h_size; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < i_size; ++j)
            {
                sum += this->w(i, j) * input[j];
            }
            sum += this->b[i];
            h[i] = leaky_relu(sum);
        }

        // ������� �������� �������� �������� ������
        std::vector<double> y_pred(o_size); // { v_next, x_next, y_next, theta_next }
        for (int i = 0; i < o_size; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < h_size; ++j)
            {
                sum += this->wo(i, j) * h[j];
            }
            sum += this->bo[i];
            y_pred[i] = leaky_relu(sum);
        }

        return y_pred;
    }

    // �������� ��������� ����
    void train(const std::vector<std::vector<double>>& data,
               const std::vector<std::vector<double>>& all_y_trues,
               const int& epochs,
               const double& t)
    {
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            for (const auto& [x, y_true] : std::views::zip(data, all_y_trues)) 
            {
                // ------------------- ������ ������ ��� �������� ���� ------------------- //

               // !!! (��� ��� ���� ���� � ���� ����������, ������� �������� ���������
               // �� ����� ����� ������� feedforward) !!!

                 // ������� �������� 240 �������� �������� ����
                std::vector<double> sum_h(h_size, 0.0);
                std::vector<double> h(h_size, 0.0);
                for (int i = 0; i < h_size; ++i) {
                    for (int j = 0; j < i_size; ++j) {
                        sum_h[i] += this->w(i, j) * x[j];
                    }
                    sum_h[i] += this->b[i];
                    h[i] = leaky_relu(sum_h[i]);
                }

                // ------------------- ������ ������ ��� ��������� ���� ------------------- //
                 // p����� �������� 4 �������� ��������� ����
                std::vector<double> sum_ypred(o_size, 0.0);
                std::vector<double> y_pred(o_size, 0.0);
                for (int i = 0; i < o_size; ++i) {
                    for (int j = 0; j < h_size; ++j) {
                        sum_ypred[i] += this->wo(i, j) * h[j];
                    }
                    sum_ypred[i] += this->bo[i];
                    y_pred[i] = leaky_relu(sum_ypred[i]);
                }

                // =================================================== ������� ����������� ������� ������ L ��� ���� ����� � �������� ============================================= //===
                // ����������� ������� ������
                std::vector<double> dL_dy_pred(o_size);
                for (int j = 0; j < o_size; ++j) {
                    dL_dy_pred[j] = 2.0 * (y_pred[j] - y_true[j]);
                }

                // ------------------- ������ ������� ����������� ��� y_pred �� wo � bo -------------------- //
                std::vector<std::vector<double>> dy_pred_dwo(o_size, std::vector<double>(h_size));
                std::vector<std::vector<double>> dy_pred_dh(o_size, std::vector<double>(h_size));
                std::vector<double> dy_pred_bo(o_size);

                for (int i = 0; i < o_size; ++i) {
                    double deriv = deriv_leaky_relu(sum_ypred[i]);
                    dy_pred_bo[i] = deriv;

                    for (int j = 0; j < h_size; ++j) {
                        dy_pred_dwo[i][j] = h[j] * deriv;
                        dy_pred_dh[i][j] = this->wo(i, j) * deriv;
                    }
                }

                // ------------------- ������ ������� ����������� ��� h �� w � b -------------------- //
                std::vector<std::vector<double>> dh_dw(h_size, std::vector<double>(i_size));
                std::vector<double> dh_db(h_size);

                for (int i = 0; i < h_size; ++i) {
                    double deriv = deriv_leaky_relu(sum_h[i]);
                    dh_db[i] = deriv;

                    for (int j = 0; j < i_size; ++j) {
                        dh_dw[i][j] = x[j] * deriv_leaky_relu(sum_h[i]);
                    }
                }

                // ================================================ ���������� ����� � �������� =============================================== //

                // ------------------- ���������� ����� �������� ���� (w) -------------------- //
                for (int i = 0; i < h_size; ++i) {
                    for (int j = 0; j < i_size; ++j) {
                        double dL_dwij = 0.0;
                        for (int k = 0; k < o_size; ++k) {
                            dL_dwij += dL_dy_pred[k] * dy_pred_dh[k][i] * dh_dw[i][j];
                        }
                        this->w(i, j) -= t * dL_dwij;
                    }
                }

                // ------------------- ���������� �������� �������� ���� (b) -------------------- //
                for (int i = 0; i < h_size; ++i) {
                    double dL_dbi = 0.0;
                    for (int k = 0; k < o_size; ++k) {
                        dL_dbi += dL_dy_pred[k] * dy_pred_dh[k][i] * dh_db[i];
                    }
                    this->b[i] -= t * dL_dbi;
                }

                // ------------------- ���������� ����� ��������� ���� (wo) -------------------- //
                for (int i = 0; i < o_size; ++i) {
                    for (int j = 0; j < h_size; ++j) {
                        this->wo(i, j) -= t * dL_dy_pred[i] * dy_pred_dwo[i][j];
                    }
                }

                // ------------------- ���������� �������� ��������� ���� (bo) -------------------- //
                for (int i = 0; i < o_size; ++i)
                {
                    this->bo[i] -= t * dL_dy_pred[i] * dy_pred_bo[i];
                }
            }

            // ����� ����� ������ 100 ����
            if (epoch % 100 == 0)
            {
                double total_loss = 0.0;
                for (size_t i = 0; i < data.size(); ++i)
                {
                    std::vector<double> y_pred = this->feedforward(data[i]);
                    total_loss += mse_loss(all_y_trues[i], y_pred);
                }
                std::cout << "����� " << epoch << " ������: " << total_loss / data.size() << std::endl;
            }
        }
    }
};

int main()
{
    setlocale(LC_ALL, "Russian");
    try
    {
        // ��������� ������� ������
        std::vector<std::vector<double>> data =
        {
            {0.0000, 0.0000, 0.1745, 60.0000, 0.0000, 2.0000},
            {0.0000, 0.0000, 0.2618, 60.0000, 0.0000, 2.0000},
            {0.0000, 0.0000, 0.3491, 60.0000, 0.0000, 2.0000},
            {114.7652, 22.8330, 0.0226, 60.4851, 2.0000, 4.0000},
            {0.0000, 0.0000, 0.4363, 60.0000, 0.0000, 2.0000},
            {110.6537, 32.9938, 0.1126, 58.7727, 2.0000, 4.0000},
            {0.0000, 0.0000, 0.5236, 60.0000, 0.0000, 2.0000},
            {105.6783, 42.7270, 0.2058, 57.0807, 2.0000, 4.0000},
            {220.6904, 46.9411, -0.1318, 50.5867, 4.0000, 6.0000},
            {0.0000, 0.0000, 0.6109, 60.0000, 0.0000, 2.0000},
            {99.8843, 51.9549, 0.3023, 55.4275, 2.0000, 4.0000},
            {208.9848, 66.5778, -0.0477, 47.7703, 4.0000, 6.0000},
            {0.0000, 0.0000, 0.6981, 60.0000, 0.0000, 2.0000},
            {93.3236, 60.6056, 0.4022, 53.8319, 2.0000, 4.0000},
            {195.5862, 85.1348, 0.0425, 44.8842, 4.0000, 6.0000},
            {276.1583, 71.3274, -0.4232, 42.8027, 6.0000, 8.0000},
            {0.0000, 0.0000, 0.7854, 60.0000, 0.0000, 2.0000},
            {86.0545, 68.6141, 0.5057, 52.3132, 2.0000, 4.0000},
            {180.6007, 102.4381, 0.1404, 41.9586, 4.0000, 6.0000},
            {255.1183, 96.2440, -0.3745, 39.1225, 6.0000, 8.0000},
            {0.0000, 0.0000, 0.8727, 60.0000, 0.0000, 2.0000},
            {78.1399, 75.9224, 0.6128, 50.8906, 2.0000, 4.0000},
            {164.1552, 118.3215, 0.2474, 39.0342, 4.0000, 6.0000},
            {231.7643, 119.3208, -0.3251, 35.1801, 6.0000, 8.0000},
            {291.3244, 79.2021, -0.7697, 41.7146, 8.0000, 10.0000},
            {0.0000, 0.0000, 0.9599, 60.0000, 0.0000, 2.0000},
            {69.6468, 82.4799, 0.7234, 49.5829, 2.0000, 4.0000},
            {146.3972, 132.6318, 0.3653, 36.1637, 4.0000, 6.0000},
            {206.2885, 140.2735, -0.2743, 30.9632, 6.0000, 8.0000},
            {259.0054, 105.4441, -0.7792, 38.2358, 8.0000, 10.0000},
            {0.0000, 0.0000, 1.0472, 60.0000, 0.0000, 2.0000},
            {60.6449, 88.2434, 0.8373, 48.4080, 2.0000, 4.0000},
            {127.4924, 145.2348, 0.4959, 33.4128, 4.0000, 6.0000},
            {178.9426, 158.8000, -0.2200, 26.4701, 6.0000, 8.0000},
            {223.7131, 128.9587, -0.8047, 34.6087, 8.0000, 10.0000},
            {0.0000, 0.0000, 1.1345, 60.0000, 0.0000, 2.0000},
            {51.2056, 93.1767, 0.9543, 47.3827, 2.0000, 4.0000},
            {107.6201, 156.0203, 0.6408, 30.8600, 4.0000, 6.0000},
            {150.0611, 174.5832, -0.1560, 21.7301, 6.0000, 8.0000},
            {185.7453, 149.6210, -0.8520, 30.8971, 8.0000, 10.0000},
            {0.0000, 0.0000, 1.2217, 60.0000, 0.0000, 2.0000},
            {41.4016, 97.2505, 1.0740, 46.5218, 2.0000, 4.0000},
            {86.9658, 164.9073, 0.8011, 28.5930, 4.0000, 6.0000},
            {120.0901, 187.3178, -0.0656, 16.8455, 6.0000, 8.0000},
            {145.6005, 167.6679, -0.9305, 27.1925, 8.0000, 10.0000},
            {172.4354, 105.9635, -1.1792, 41.1872, 10.0000, 12.0000},
            {0.0000, 0.0000, 1.3090, 60.0000, 0.0000, 2.0000},
            {31.3060, 100.4417, 1.1961, 45.8382, 2.0000, 4.0000},
            {65.7125, 171.8438, 0.9765, 26.7034, 4.0000, 6.0000}
        };

        // �������� �������� ������ ��� ��������
        std::vector<std::vector<double>> all_y_trues =
        {
            {-0.1489, 63.9037, 120.2571, 1.5649},
            {-0.0645, 62.2008, 117.9758, 12.3277},
            {0.0226, 60.4851, 114.7652, 22.8330},
            {-0.2854, 55.9299, 238.6983, 5.1413},
            {0.1126, 58.7727, 110.6537, 32.9938},
            {-0.2107, 53.3116, 230.6171, 26.4025},
            {0.2058, 57.0807, 105.6783, 42.7270},
            {-0.1318, 50.5867, 220.6904, 46.9411},
            {-0.5189, 49.4381, 310.7485, 17.0501},
            {0.3023, 55.4275, 99.8843, 51.9549},
            {-0.0477, 47.7703, 208.9848, 66.5778},
            {-0.4714, 46.2361, 294.7381, 44.8421},
            {0.4022, 53.8319, 93.3236, 60.6056},
            {0.0425, 44.8842, 195.5862, 85.1348},
            {-0.4232, 42.8027, 276.1583, 71.3274},
            {-0.7817, 48.1076, 346.3419, 19.6972},
            {0.5057, 52.3132, 86.0545, 68.6141},
            {0.1404, 41.9586, 180.6007, 102.4381},
            {-0.3745, 39.1225, 255.1183, 96.2440},
            {-0.7716, 45.0101, 320.4788, 50.5100},
            {0.6128, 50.8906, 78.1399, 75.9224},
            {0.2474, 39.0342, 164.1552, 118.3215},
            {-0.3251, 35.1801, 231.7643, 119.3208},
            {-0.7697, 41.7146, 291.3244, 79.2021},
            {-1.0243, 50.7058, 344.7620, 5.4666},
            {0.7234, 49.5829, 69.6468, 82.4799},
            {0.3653, 36.1637, 146.3972, 132.6318},
            {-0.2743, 30.9632, 206.2885, 140.2735},
            {-0.7792, 38.2358, 259.0054, 105.4441},
            {-1.0436, 48.3190, 307.2276, 35.3196},
            {0.8373, 48.4080, 60.6449, 88.2434},
            {0.4959, 33.4128, 127.4924, 145.2348},
            {-0.2200, 26.4701, 178.9426, 158.8000},
            {-0.8047, 34.6087, 223.7131, 128.9587},
            {-1.0744, 45.9048, 265.7489, 62.0363},
            {0.9543, 47.3827, 51.2056, 93.1767},
            {0.6408, 30.8600, 107.6201, 156.0203},
            {-0.1560, 21.7301, 150.0611, 174.5832},
            {-0.8520, 30.8971, 185.7453, 149.6210},
            {-1.1186, 43.5120, 220.6366, 85.4884},
            {1.0740, 46.5218, 41.4016, 97.2505},
            {0.8011, 28.5930, 86.9658, 164.9073},
            {-0.0656, 16.8455, 120.0901, 187.3178},
            {-0.9305, 27.1925, 145.6005, 167.6679},
            {-1.1792, 41.1872, 172.4354, 105.9635},
            {-1.3038, 52.5758, 200.1806, 15.1892},
            {1.1961, 45.8382, 31.3060, 100.4417},
            {0.9765, 26.7034, 65.7125, 171.8438},
            {0.0900, 12.0621, 89.5995, 196.7942}
        };

        // ������������ ������� � �������� ������
        Normalization_Params in_true_params = normalize_to_zero_one(data);
        Normalization_Params y_true_params = normalize_to_zero_one(all_y_trues);

        // �������� ������� � �������� ������� �� ������ ��������� ������
        std::vector<double> test_input = { 0.0000, 0.0000, 0.1745, 60.0000, 0.0000, 2.0000 };
        std::vector<double> test_true = { -0.1489, 63.9037, 120.2571, 1.5649 };

        // �������� � �������� ��������� ����
        My_First_Neuron_Network network;
        network.train(data, all_y_trues, 10000, 0.05);

        // ��������� � ����� ������������
        std::vector<double> normalized_input = normalize_vector(test_input, in_true_params);
        std::vector<double> normalized_prediction = network.feedforward(normalized_input);
        std::vector<double> final_prediction = denormalize_vector(normalized_prediction, y_true_params);

        // ����� �������� ������ � ������������
        std::cout << "����������� ������:\n";
        for (double val : test_input)
        {
            std::cout << val << " ";
        }
        std::cout << "\n��������� ������������:\n";
        for (double val : test_true)
        {
            std::cout << val << " ";
        }
        std::cout << "\n������������ ���������:\n";
        for (double val : final_prediction)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "������: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}