#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>

// хранение входных данных в структуре чтобы не было двумерных массивов
struct Y
{
    double x, y, v, theta;

    Y(double x, double y, double v, double theta) : x(x), y(y), v(v), theta(theta) {}

};

// Класс для хранения выходных данных (y_i+1, x_i+1, theta_i+1, v_i+1)
class Y_next {
public:
    double x, y, v, theta;
    Y_next(double x, double y, double v, double theta) : x(x), y(y), v(v), theta(theta) {}
};

// функция активации - сигмоида f(x) = 1 / (1 + e^-x)
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// производная сигмоиды
double deriv_sigmoid(double x) {
    double fx = sigmoid(x);
    return fx * (1 - fx);
}

// функция потерь считает среднюю квадратичную ошибку (mean squared error, MSE):
// n – это количество измерений, сколько обучающих наборов данных
// y_true – истинное значение переменной("правильный ответ") из обучающего набора данных
// y_pred – предсказанное значение переменной. Это то, что выдаст моя нейронная сеть
// (ytrue - ypred)^2 называется квадратичной ошибкой
double mse_loss(const std::vector<Y>& y_true, const std::vector<Y>& y_pred) {
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
    // Веса и смещения для нейронной сети
    double w1[6], w2[6], w3[6], w4[6], w5[6], w6[6];
    double w_out1[6], w_out2[6], w_out3[6], w_out4[6];
    double b1, b2, b3, b4, b5, b6, b_out1, b_out2, b_out3, b_out4;


    My_First_Neuron_Network() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        // Инициализация весов для 6 скрытых нейронов
        for (int i = 0; i < 6; ++i) {
            w1[i] = dis(gen);
            w2[i] = dis(gen);
            w3[i] = dis(gen);
            w4[i] = dis(gen);
            w5[i] = dis(gen);
            w6[i] = dis(gen);
        }

        // Инициализация весов для выходного слоя
        for (int i = 0; i < 6; ++i) {
            w_out1[i] = dis(gen);
            w_out2[i] = dis(gen);
            w_out3[i] = dis(gen);
            w_out4[i] = dis(gen);
        }

        // Инициализация смещений
        b1 = dis(gen); b2 = dis(gen); b3 = dis(gen); b4 = dis(gen); b5 = dis(gen); b6 = dis(gen);
        b_out1 = dis(gen); b_out2 = dis(gen); b_out3 = dis(gen); b_out4 = dis(gen);
    }

    // Функция активации: Сигмоида
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    // Производная функции активации
    double deriv_sigmoid(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    Y_next feedforward(const Y& x, double t_i, double t_next)
    {
        // Прямой проход для получения предсказаний
        double h[6];
        h[0] = sigmoid(w1[0] * x.x + w1[1] * x.y + w1[2] * x.v + w1[3] * x.theta + w1[4] * t_i + w1[5] * t_next + b1);
        h[1] = sigmoid(w2[0] * x.x + w2[1] * x.y + w2[2] * x.v + w2[3] * x.theta + w2[4] * t_i + w2[5] * t_next + b2);
        h[2] = sigmoid(w3[0] * x.x + w3[1] * x.y + w3[2] * x.v + w3[3] * x.theta + w3[4] * t_i + w3[5] * t_next + b3);
        h[3] = sigmoid(w4[0] * x.x + w4[1] * x.y + w4[2] * x.v + w4[3] * x.theta + w4[4] * t_i + w4[5] * t_next + b4);
        h[4] = sigmoid(w5[0] * x.x + w5[1] * x.y + w5[2] * x.v + w5[3] * x.theta + w5[4] * t_i + w5[5] * t_next + b5);
        h[5] = sigmoid(w6[0] * x.x + w6[1] * x.y + w6[2] * x.v + w6[3] * x.theta + w6[4] * t_i + w6[5] * t_next + b6);

        // Выходные данные
        double v_next = sigmoid(w_out1[0] * h[0] + w_out1[1] * h[1] + w_out1[2] * h[2] + w_out1[3] * h[3] + w_out1[4] * h[4] + w_out1[5] * h[5] + b_out1);
        double x_next = sigmoid(w_out2[0] * h[0] + w_out2[1] * h[1] + w_out2[2] * h[2] + w_out2[3] * h[3] + w_out2[4] * h[4] + w_out2[5] * h[5] + b_out2);
        double y_next = sigmoid(w_out3[0] * h[0] + w_out3[1] * h[1] + w_out3[2] * h[2] + w_out3[3] * h[3] + w_out3[4] * h[4] + w_out3[5] * h[5] + b_out3);
        double theta_next = sigmoid(w_out4[0] * h[0] + w_out4[1] * h[1] + w_out4[2] * h[2] + w_out4[3] * h[3] + w_out4[4] * h[4] + w_out4[5] * h[5] + b_out4);

        return Y_next (v_next, x_next, y_next, theta_next);
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

