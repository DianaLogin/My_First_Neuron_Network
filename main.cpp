#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include "Matrix.h"
#include "Vector.h"

// функция активации - сигмоида f(x) = 1 / (1 + e^-x)
double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

// производная сигмоиды
double deriv_sigmoid(double x)
{
    double fx = sigmoid(x);
    return fx * (1 - fx);
}

// функция потерь считает среднюю квадратичную ошибку (mean squared error, MSE):
// n – это количество выходных значений (в данном случае 4 на каждый пример)
// y_true – истинное значение переменной ("правильный ответ") из обучающего набора данных
// y_pred – предсказанное значение переменной. Это то, что выдаёт моя нейронная сеть
// (y_true - y_pred)^2 называется квадратичной ошибкой
double mse_loss(const Vector<double>& y_true, const Vector<double>& y_pred)
{
    double sum = 0.0;
    size_t n = 4;
    for (size_t i = 0; i < n; ++i)
    {
        sum += std::pow(y_true[i] - y_pred[i], 2);
    }
    return sum / n;
}


class My_First_Neuron_Network
{
    // состоит из:
    // 6 входных нейронов theta_i, y_i, x_i, t_i, t_i+1, v_i
    // 6 скрытых нейронов 1 скрытого слоя
    // 4 выходных нейронов y_i+1, x_i+1, theta_i+1, v_i+1
private:
    Matrix<double, 6, 6> w;
    /*{
        {w1, w2, w3, w4, w5, w6}, // при нейроне h1
        {w7, w8, w9, w10, w11, w12}, // при нейроне h2
        {w13, w14, w15, w16, w17, w18}, // при нейроне h3
        {w19, w20, w21, w22, w23, w24}, // при нейроне h4
        {w25, w26, w27, w28, w29, w30}, // при нейроне h5
        {w31, w32, w33, w34, w35, w36} // при нейроне h6
    };*/

    Matrix<double, 4, 6> wo;
    /*{
        {wo1, w02, w03, w04, w05, w06}, // при нейроне o1
        {w07, w08, w09, w010, w011, w012}, // при нейроне o1
        {w013, w014, w015, w016, w017, w018}, // при нейроне o1
        {w019, w020, w021, w022, w023, w024} // при нейроне o1
    };*/

    Vector<double> b; // {b1,b2,b3,b4,b5,b6}
    Vector<double> bo; // {b01,b02,b03,b04}


public:
    // только так получится задать размер векторам, чтобы можно было заполнить их значениями
    // в полях класса почему-то не получается так написать Vector<double> b(6)
    My_First_Neuron_Network() : b(6), bo(4)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        // инициализация весов для 6 скрытых нейронов
        for (int i = 0; i < 6; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                w(i, j) = dis(gen);
            }
        }

        // инициализация весов для выходного слоя
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                wo(i, j) = dis(gen);

            }
        }

        // инициализация смещений от входного слоя к скрытому слою
        for (int i = 0; i < 6; ++i)
        {
            b[i] = dis(gen);
        }

        // инициализация смещений от скрытого слоя к выходному слою
        for (int i = 0; i < 4; ++i)
        {
            bo[i] = dis(gen);
        }
    }

    // прямой проход для получения предсказаний
    Vector<double> feedforward(Vector<double>& input)
    {
        
        // рассчёт значения нейронов скрытого слоя 
        Vector<double> h(6); // = { h1, h2, h3, h4, h5, h6 };
        for (int i = 0; i < 6; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < 6; ++j)
            {
                sum += this->w(i, j) * input[j];
            }
            sum += this->b[i];
            h[i] = sigmoid(sum);
        }

        // рассчёт значения нейронов выходных данных
        Vector<double> y_pred(4); // { v_next, x_next, y_next, theta_next }
        for (int i = 0; i < 4; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < 6; ++j)
            {
                sum += this->wo(i, j) * h[j];
            }
            sum += this->bo[i];
            y_pred[i] = sigmoid(sum);
        }

        return y_pred;
    }

    void train(const std::vector<Vector<double>>& data, const std::vector<Vector<double>>& all_y_trues, const int& epochs, const double& t)
    {
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            for (auto [x, y_true] : std::views::zip(data, all_y_trues)) // data, all_y_trues состоят из 15 наборов обучающмх данных
            {
                // --- прямой проход для скрытого слоя ---

               // !!! (они все чуть ниже в коде пригодятся, поэтому пришлось расписать
               // не через вызов функции feedforward) !!!

                 // расчёт значений 6 нейронов скрытого слоя
                Vector<double> sum_h(6);
                Vector<double> h(6);
                for (int i = 0; i < 6; ++i)
                {
                    double sum = 0.0;
                    for (int j = 0; j < 6; ++j)
                    {
                        sum += this->w(i, j) * x[j];
                    }
                    sum += this->b[i];
                    sum_h[i] = sum;

                    h[i] = sigmoid(sum);
                }

                // --- прямой проход для выходного слоя ---
                // pасчёт значений 4 нейронов выходного слоя
                Vector<double> sum_ypred(4);
                Vector<double> y_pred(4);
                for (int i = 0; i < 4; ++i)
                {
                    double sum = 0.0;
                    for (int j = 0; j < 4; ++j)
                    {
                        sum += this->wo(i, j) * x[j];
                    }
                    sum += this->bo[i];
                    sum_ypred[i] = sum;

                    y_pred[i] = sigmoid(sum);
                }


        // =================================================== Частные производные функции потерь L для всех весов и смещений ============================================= //===
                Vector<double> dL_dy_pred(4);
                for (int j = 0; j < 4; ++j)
                {
                    dL_dy_pred[j] = 2.0 * (y_pred[j] - y_true[j]);  // Производная функции потерь
                }


        // ------------------- расчёт частных производных для y_pred по wo и bo -------------------- //
                // для нейрона y_pred[0] 
                Vector<double> dy_pred_dwo_0(6);
                Vector<double> dy_pred_dh_0(6);
                for (int i = 0; i < 6; ++i)
                {
                    dy_pred_dwo_0[i] = h[i] * deriv_sigmoid(sum_ypred[0]);
                    dy_pred_dh_0[i] = this->wo(0 , i ) * deriv_sigmoid(sum_ypred[0]);
                }
               
                // для нейрона y_pred[1]
                Vector<double> dy_pred_dwo_1(6);
                Vector<double> dy_pred_dh_1(6);
                for (int i = 0; i < 6; ++i)
                {
                    dy_pred_dwo_1[i] = h[i] * deriv_sigmoid(sum_ypred[1]);
                    dy_pred_dh_1[i] = this->wo(1 , i ) * deriv_sigmoid(sum_ypred[1]);
                }
               
                // для нейрона y_pred[2] 
                Vector<double> dy_pred_dwo_2(6);
                Vector<double> dy_pred_dh3(6);
                for (int i = 0; i < 6; ++i)
                {
                    dy_pred_dwo_2[i] = h[i] * deriv_sigmoid(sum_ypred[2]);
                    dy_pred_dh3[i] = this->wo(2 , i ) * deriv_sigmoid(sum_ypred[2]);
                }
               
                // для нейрона y_pred[3]
                Vector<double> dy_pred_dwo_3(6);
                Vector<double> dy_pred_dh4(6);
                for (int i = 0; i < 6; ++i)
                {
                    dy_pred_dwo_3[i] = h[i] * deriv_sigmoid(sum_ypred[3]);
                    dy_pred_dh4[i] = this->wo(3 , i ) * deriv_sigmoid(sum_ypred[3]);
                }


               // смещения для скрытого -> выходного слоя
                Vector<double> dy_pred_bo(4);
                for (int i = 0; i < 4; ++i)
                {
                    dy_pred_bo[i] = deriv_sigmoid(sum_ypred[i]);
                }

        // ------------------- расчёт частных производных для h по w и b -------------------- //       
                // для нейрона h[0]
                Vector<double> dh0_dw_0(6);
                for (int i = 0; i < 6; ++i)
                {
                    dh0_dw_0[i] = x[i] * deriv_sigmoid(sum_h[0]);
                }

                // для нейрона h[1]
                Vector<double> dh1_dw_1(6);
                for (int i = 0; i < 6; ++i)
                {
                    dh1_dw_1[i] = x[i] * deriv_sigmoid(sum_h[1]);
                }

                // для нейрона h[2]
                Vector<double> dh2_dw_2(6);
                for (int i = 0; i < 6; ++i)
                {
                    dh2_dw_2[i] = x[i] * deriv_sigmoid(sum_h[2]);
                }

                // для нейрона h[3]
                Vector<double> dh3_dw_3(6);
                for (int i = 0; i < 6; ++i)
                {
                    dh3_dw_3[i] = x[i] * deriv_sigmoid(sum_h[3]);
                }

                // для нейрона h[4]
                Vector<double> dh4_dw_4(6);
                for (int i = 0; i < 6; ++i)
                {
                    dh4_dw_4[i] = x[i] * deriv_sigmoid(sum_h[4]);
                }

                // для нейрона h[5]
                Vector<double> dh5_dw_5(6);
                for (int i = 0; i < 6; ++i)
                {
                    dh5_dw_5[i] = x[i] * deriv_sigmoid(sum_h[5]);
                }


                // смещения для входного -> скрытого слоя
                Vector<double> dh__db(6);
                for (int i = 0; i < 6; ++i)
                {
                    dh__db[i] = deriv_sigmoid(sum_h[i]);
                }

               

        // ================================================ Обновление весов и смещений =============================================== //
                // для нейрона h[0]
                for (int i = 0; i < 6; ++i)
                {
                    this->w(0, i) -= t * dL_dy_pred *dy_pred_dh;
                }

            }
        }
    }
};

int main()
{
    /* Matrix<int, 2, 3> m = { {1,2,3}, {4,5,6}};
     std::cout << m(0,0);*/

    My_First_Neuron_Network nw;
    
}
