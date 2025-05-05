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
// n – это количество измерений, сколько обучающих наборов данных
// y_true – истинное значение переменной("правильный ответ") из обучающего набора данных
// y_pred – предсказанное значение переменной. Это то, что выдаст моя нейронная сеть
// (ytrue - ypred)^2 называется квадратичной ошибкой
double mse_loss(const std::vector<Y>& y_true, const std::vector<Y>& y_pred) 
{
    if (y_true.size() != y_pred.size()) {
        std::cerr << "Размеры y_true и y_pred не совпадают!" << std::endl;
        return -1;
    }

    double sum = 0.0;
    size_t n = y_true.size();
    for (size_t i = 0; i < n; ++i) {
        sum += std::pow(y_true[i].x - y_pred[i].x, 2) +
            std::pow(y_true[i].y - y_pred[i].y, 2) +
            std::pow(y_true[i].v - y_pred[i].v, 2) +
            std::pow(y_true[i].theta - y_pred[i].theta, 2);
    }
    return sum / n;
}

class My_First_Neuron_Network
{
public:

    My_First_Neuron_Network() 
    {
        // Состоит из:
        // 6 входных нейронов theta_i, y_i, x_i, t_i, t_i+1, v_i
        // 6 скрытых нейронов 1 скрытого слоя
        // 4 выходных нейронов y_i+1, x_i+1, theta_i+1, v_i+1


        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);



        // Инициализация весов для 6 скрытых нейронов
        Matrix<double, 6, 6> w;/*=
        {
            {w1, w2, w3, w4, w5, w6},
            {w7, w8, w9, w10, w11, w12},
            {w13, w14, w15, w16, w17, w18},
            {w19, w20, w21, w22, w23, w24},
            {w25, w26, w27, w28, w29, w30},
            {w31, w32, w33, w34, w35, w36}
        };*/
        for (int i = 0; i < 6; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                w(i, j) = dis(gen);

            }
        }

        // Инициализация весов для выходного слоя
        Matrix<double, 4, 6> wo;/*=
         {
            {wo1, w02, w03, w04, w05, w06},
            {w07, w08, w09, w010, w011, w012},
            {w013, w014, w015, w016, w017, w018},
            {w019, w020, w021, w022, w023, w024}
        };*/
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                wo(i, j) = dis(gen);

            }
        }

        // Инициализация смещений от входного слоя к скрытому слою
        Vector<double> b(6);
        for (int i = 0; i < 6; ++i)
        {
            b[i] = dis(gen);
        }

        // Инициализация смещений от скрытого слоя к выходному слою
        Vector<double> bo(4);
        for (int i = 0; i < 4; ++i)
        {
            bo[i] = dis(gen);
        }

    }

    Vector<double> feedforward(Vector<double>& in, double t_i, double t_next,
                               Vector<double>& b, Vector<double>& bo,
                               Matrix<double,6,6>& w, Matrix<double, 4,6>& wo)
    {
        Vector<double> h(6); // = { h1, h2, h3, h4, h5, h6 };
        for (int i = 0; i < 6; ++i)
        {
            h[i] = sigmoid(w(i,j)*in[i] + w(i,j)
        }
        
        // Прямой проход для получения предсказаний
       /* double h[6];
        h[0] = sigmoid(w1[0] * x.x + w1[1] * x.y + w1[2] * x.v + w1[3] * x.theta + w1[4] * t_i + w1[5] * t_next + b1);
        h[1] = sigmoid(w2[0] * x.x + w2[1] * x.y + w2[2] * x.v + w2[3] * x.theta + w2[4] * t_i + w2[5] * t_next + b2);
        h[2] = sigmoid(w3[0] * x.x + w3[1] * x.y + w3[2] * x.v + w3[3] * x.theta + w3[4] * t_i + w3[5] * t_next + b3);
        h[3] = sigmoid(w4[0] * x.x + w4[1] * x.y + w4[2] * x.v + w4[3] * x.theta + w4[4] * t_i + w4[5] * t_next + b4);
        h[4] = sigmoid(w5[0] * x.x + w5[1] * x.y + w5[2] * x.v + w5[3] * x.theta + w5[4] * t_i + w5[5] * t_next + b5);
        h[5] = sigmoid(w6[0] * x.x + w6[1] * x.y + w6[2] * x.v + w6[3] * x.theta + w6[4] * t_i + w6[5] * t_next + b6);*/
        
       
        

        // Выходные данные
        double v_next = sigmoid(w_out1[0] * h[0] + w_out1[1] * h[1] + w_out1[2] * h[2] + w_out1[3] * h[3] + w_out1[4] * h[4] + w_out1[5] * h[5] + b_out1);
        double x_next = sigmoid(w_out2[0] * h[0] + w_out2[1] * h[1] + w_out2[2] * h[2] + w_out2[3] * h[3] + w_out2[4] * h[4] + w_out2[5] * h[5] + b_out2);
        double y_next = sigmoid(w_out3[0] * h[0] + w_out3[1] * h[1] + w_out3[2] * h[2] + w_out3[3] * h[3] + w_out3[4] * h[4] + w_out3[5] * h[5] + b_out3);
        double theta_next = sigmoid(w_out4[0] * h[0] + w_out4[1] * h[1] + w_out4[2] * h[2] + w_out4[3] * h[3] + w_out4[4] * h[4] + w_out4[5] * h[5] + b_out4);

        return y_pred = { v_next, x_next, y_next, theta_next };
    }

    void backpropagate(const Y& x, double t_i, double t_next, const Y& y_true, double learning_rate)
    {
        // Прямой проход для получения предсказаний
        Y y_pred = feedforward(x, t_i, t_next);

        // Вычисление ошибки (производной от функции потерь по выходным данным)
        double delta_o1 = 2 * (y_pred.x - y_true.x) * deriv_sigmoid(y_pred.x);
        double delta_o2 = 2 * (y_pred.y - y_true.y) * deriv_sigmoid(y_pred.y);
        double delta_o3 = 2 * (y_pred.v - y_true.v) * deriv_sigmoid(y_pred.v);
        double delta_o4 = 2 * (y_pred.theta - y_true.theta) * deriv_sigmoid(y_pred.theta);

        // Обновление весов и смещений для выходного слоя
        for (int i = 0; i < 6; ++i) 
        {
            w_out1[i] -= learning_rate * delta_o1 * w_out1[i];
            w_out2[i] -= learning_rate * delta_o2 * w_out2[i];
            w_out3[i] -= learning_rate * delta_o3 * w_out3[i];
            w_out4[i] -= learning_rate * delta_o4 * w_out4[i];
        }

    }

};

