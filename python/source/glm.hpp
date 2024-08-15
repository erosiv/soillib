#ifndef SOILLIB_PYTHON_GLM
#define SOILLIB_PYTHON_GLM

//! GLM Vector and Matrix Type Casters
//!
//! Note that with this, there is no requirement
//! to expose the types through pybind, as they
//! are directly cast to and from numpy types.
//!
//! Converters for matrices assume col-major
//! storage order for GLM. Will break otherwise.
//!
//! Based on work by Patrik Huber 2016,
//! adapted for latest Pybind11 / C++23 

#include "pybind11/numpy.h"
#include "glm/gtc/type_ptr.hpp"

#include <iostream>
#include <cstddef>

namespace pybind11 {
namespace detail {

template<typename T, glm::precision P>
struct type_caster<glm::tvec2<T, P>> {

	typedef glm::tvec2<T, P> vector;
	typedef T scalar;
	static constexpr size_t num_elements = 2;

	PYBIND11_TYPE_CASTER(vector, const_name("tvec2"));

	bool load(handle src, bool){

		array_t<scalar> buf(src, true);
		if(!buf.check())
			return false;

		if(buf.ndim() != 1)
			return false;

		if(buf.shape(0) != num_elements)
			return false;
	
		if(buf.strides(0) != sizeof(scalar)){
			std::cout << "An array with non-standard strides is given. Please pass a contiguous array." << std::endl;
			return false;
		}

		value = glm::make_vec2(buf.mutable_data()); // make_vec* copies the data (unnecessarily)
		return true;

	}

	static handle cast(const vector& src, return_value_policy, handle){
		return array(
			num_elements,
			glm::value_ptr(src)
		).release();
	}

	// Specifies the doc-string for the type in Python:
	// PYBIND11_TYPE_CASTER(vector_type, _("numpy.ndarray[") + npy_format_descriptor<Scalar>::name() +
	// 	_("[") + elements() + _("]]"));

// protected:
// 	template <typename T = vector_type>
// 	static std::string elements() { return _(std::to_string(num_elements).c_str()); }
};

template<typename T, glm::precision P>
struct type_caster<glm::tvec3<T, P>> {
	
	typedef glm::tvec3<T, P> vector;
	typedef T scalar;
	static constexpr std::size_t num_elements = 3;

	PYBIND11_TYPE_CASTER(vector, const_name("tvec3"));

	bool load(handle src, bool){

		array_t<scalar> buf(src, true);
		if(!buf.check())
			return false;

		if(buf.ndim() != 1)
			return false;

		if(buf.shape(0) != num_elements)
			return false;
	
		if(buf.strides(0) != sizeof(scalar)){
			std::cout << "An array with non-standard strides is given. Please pass a contiguous array." << std::endl;
			return false;
		}

		value = glm::make_vec3(buf.mutable_data()); // make_vec* copies the data (unnecessarily)
		return true;

	}

	static handle cast(const vector& src, return_value_policy, handle){
		return array(
			num_elements,
			glm::value_ptr(src)
		).release();
	}

/*
	// Specifies the doc-string for the type in Python:
	PYBIND11_TYPE_CASTER(vector_type, _("numpy.ndarray[") + npy_format_descriptor<Scalar>::name() +
		_("[") + elements() + _("]]"));

protected:
	template <typename T = vector_type>
	static PYBIND11_DESCR elements() { return _(std::to_string(num_elements).c_str()); }
*/
};

template<typename T, glm::precision P>
struct type_caster<glm::tvec4<T, P>> {

	typedef glm::tvec4<T, P> vector;
	typedef T scalar;
	static constexpr std::size_t num_elements = 4;

	PYBIND11_TYPE_CASTER(vector, const_name("tvec4"));

	bool load(handle src, bool){

		array_t<scalar> buf(src, true);
		if(!buf.check())
			return false;

		if(buf.ndim() != 1)
			return false;

		if(buf.shape(0) != num_elements)
			return false;
	
		if(buf.strides(0) != sizeof(scalar)){
			std::cout << "An array with non-standard strides is given. Please pass a contiguous array." << std::endl;
			return false;
		}

		value = glm::make_vec4(buf.mutable_data()); // make_vec* copies the data (unnecessarily)
		return true;

	}

	static handle cast(const vector& src, return_value_policy, handle) {
		return array(
			num_elements,
			glm::value_ptr(src)
		).release();
	}

/*
	// Specifies the doc-string for the type in Python:
	PYBIND11_TYPE_CASTER(vector_type, _("numpy.ndarray[") + npy_format_descriptor<Scalar>::name() +
		_("[") + elements() + _("]]"));

protected:
	template <typename T = vector_type>
	static PYBIND11_DESCR elements() { return _(std::to_string(num_elements).c_str()); }
*/
};

/*
template<typename T, glm::precision P>
struct type_caster<glm::tmat4x4<T, P>>
{
	using matrix_type = glm::tmat4x4<T, P>;
	typedef typename T Scalar;
	static constexpr std::size_t num_rows = 4;
	static constexpr std::size_t num_cols = 4;

	bool load(handle src, bool)
	{
		array_t<Scalar> buf(src, true);
		if (!buf.check())
			return false;

		if (buf.ndim() == 2) // a 2-dimensional matrix
		{
			if (buf.shape(0) != num_rows || buf.shape(1) != num_cols) {
				return false; // not a 4x4 matrix
			}
			if (buf.strides(0) / sizeof(Scalar) != num_cols || buf.strides(1) != sizeof(Scalar))
			{
				std::cout << "An array with non-standard strides is given. Please pass a contiguous array." << std::endl;
				return false;
			}
			// What we get from Python is laid out in row-major memory order, while GLM's
			// storage is col-major, thus, we transpose.
			value = glm::transpose(glm::make_mat4x4(buf.mutable_data())); // make_mat*() copies the data (unnecessarily)
		}
		else { // buf.ndim() != 2
			return false;
		}
		return true;
	}

	static handle cast(const matrix_type& src, return_value_policy, handle)
	{
		return array(
			{ num_rows, num_cols }, // shape
			{ sizeof(Scalar), sizeof(Scalar) * num_rows }, // strides - flip the row/col layout!
			glm::value_ptr(src)                            // data
		).release();
	}

	// Specifies the doc-string for the type in Python:
	PYBIND11_TYPE_CASTER(matrix_type, _("numpy.ndarray[") + npy_format_descriptor<Scalar>::name() +
		_("[") + rows() + _(", ") + cols() + _("]]"));

protected:
	template <typename T = matrix_type>
	static PYBIND11_DESCR rows() { return _(std::to_string(num_rows).c_str()); }
	template <typename T = matrix_type>
	static PYBIND11_DESCR cols() { return _(std::to_string(num_cols).c_str()); }
};
*/

}	// end of namespace detail
}	// end of namespace pybind11

#endif