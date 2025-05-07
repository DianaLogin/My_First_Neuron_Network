#pragma once
#include <initializer_list>
#include <iostream>
#include <memory>
#include <ranges>

template<typename T>
class Vector
{
private:
	size_t size;
	T* data;

public:
	Vector() : size(0), data(nullptr) {}
	Vector(const std::initializer_list<T>& initList) : size(initList.size())
	{
		data = new T[size];

		for (auto [i, value] : initList | std::views::enumerate)
		{
			data[i] = value;
		}
	}

	Vector(size_t data_size) : size(data_size)
	{
		data = new T[size];
	}
	Vector(const Vector& other) : size(other.size)
	{
		data = nullptr;

		if (other.size > 0)
		{
			data = new T[other.size];
			//           откуда, куда и сколько байт копируетс€
			std::memcpy(data, other.data, other.size * sizeof(T));
		}
	}


	// доступ по индексу
	T& operator[](size_t idx)
	{
		if (idx >= size)
		{
			throw std::out_of_range("»ндекс выходит за гранцы массива!");
		}
		return data[idx];
	}
	const T& operator[] (size_t idx) const
	{
		if (idx >= size)
		{
			throw std::out_of_range("»ндекс выходит за гранцы массива!");
		}
		return data[idx];
	}



	const Vector& operator= (const Vector& rhs)
	{

		if (this == &rhs)
			return *this;

		delete[] data;

		size = rhs.size;
		data = new T[size];
		for (size_t i = 0; i < size; ++i)
		{
			data[i] = rhs.data[i];
		}

		return *this;
	}

	const Vector& operator=(Vector&& rhs) noexcept
	{
		if (this == &rhs)
			return *this;

		delete[] data;

		data = rhs.data;
		size = rhs.size;

		rhs.data = nullptr;
		rhs.size = 0;

		return *this;
	}

	size_t getSize() const
	{
		return size;
	}

	~Vector()
	{
		delete[] data;
	}
};

template<typename T>
std::ostream& operator<<(std::ostream& out, const Vector<T>& arr)
{
	out << "{";
	for (size_t i = 0; i < arr.getSize(); ++i)
	{
		out << arr[i];
		if (i < arr.getSize() - 1)
		{
			out << ",";
		}
	}
	out << "}";
	return out;
}

template <typename T>
Vector<T> operator+ (const Vector<T>& v_1, const Vector<T>& v_2)
{
	if (v_1.getSize() != v_2.getSize())
	{
		throw std::invalid_argument("–азмеры складываемых векторов не совпадают!");
	}

	Vector<T> sum(v_1.getSize());

	for (size_t i = 0; i < v_1.getSize(); ++i)
	{
		sum[i] = v_1[i] + v_2[i];
	}

	return sum;
}

template <typename T>
Vector<T> operator- (const Vector<T>& v_1, const Vector<T>& v_2)
{
	if (v_1.getSize() != v_2.getSize())
	{
		throw std::invalid_argument("–азмеры вычитаемых векторов не совпадают!");
	}

	Vector<T> sum(v_1.getSize());

	for (size_t i = 0; i < v_1.getSize(); ++i)
	{
		sum[i] = v_1[i] - v_2[i];
	}

	return sum;
}

template <typename T>
T operator* (const Vector<T>& v_1, const Vector<T>& v_2)
{
	if (v_1.getSize() != v_2.getSize())
	{
		throw std::invalid_argument("–азмеры умножаемых векторов не совпадают!");
	}

	T sum = 0;
	for (size_t i = 0; i < v_1.getSize(); ++i)
	{
		sum += v_1[i] * v_2[i];
	}

	return sum;
}

template <typename T>
Vector<T> operator* (const T& scalar, const Vector<T>& v)
{
	Vector<T> res(v.getSize());
	for (size_t i = 0; i < v.getSize(); ++i)
	{
		res[i]  = scalar * v[i];
	}

	return res;
}