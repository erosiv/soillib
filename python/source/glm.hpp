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

#include <nanobind/stl/array.h>
#include "glm/gtc/type_ptr.hpp"

// #include "pybind11/numpy.h"

#include <iostream>
#include <cstddef>

namespace nanobind {
namespace detail {

/*

template<typename T, glm::precision P>
struct type_caster<glm::tvec2<T, P>> {

	typedef glm::tvec2<T, P> vector;
	typedef T scalar;
	static constexpr size_t num_elements = 2;

	bool load(handle src, bool){

		array_t<scalar> buf = array_t<scalar>::ensure(src);
		if(!buf)
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

	PYBIND11_TYPE_CASTER(vector, const_name("tvec2"));

};

template<typename T, glm::precision P>
struct type_caster<glm::tvec3<T, P>> {
	
	typedef glm::tvec3<T, P> vector;
	typedef T scalar;
	static constexpr std::size_t num_elements = 3;


	bool load(handle src, bool){

		array_t<scalar> buf = array_t<scalar>::ensure(src);
		if(!buf)
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

	PYBIND11_TYPE_CASTER(vector, const_name("tvec3"));

};

template<typename T, glm::precision P>
struct type_caster<glm::tvec4<T, P>> {

	typedef glm::tvec4<T, P> vector;
	typedef T scalar;
	static constexpr std::size_t num_elements = 4;

	bool load(handle src, bool){

		array_t<scalar> buf = array_t<scalar>::ensure(src);
		if(!buf)
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

	PYBIND11_TYPE_CASTER(vector, const_name("tvec4"));

};

*/


}	// end of namespace detail

/*
template<size_t N, typename T>
using vec = glm::vec<N, T, glm::qualifier::packed_highp>;

template<size_t N, typename T>
struct format_descriptor_ {
	static std::string format(){
		using namespace detail;
		static constexpr auto extents = const_name("(") + const_name<N>() + const_name(")");
		return extents.text + format_descriptor<remove_all_extents_t<T>>::format();
	}
};

#define glm_descriptor(N, T) template<> struct format_descriptor<vec<N, T>>: format_descriptor_<N, T>{}

glm_descriptor(1, float);
glm_descriptor(2, float);
glm_descriptor(3, float);
glm_descriptor(4, float);

glm_descriptor(1, double);
glm_descriptor(2, double);
glm_descriptor(3, double);
glm_descriptor(4, double);

glm_descriptor(1, int);
glm_descriptor(2, int);
glm_descriptor(3, int);
glm_descriptor(4, int);

*/


}

#endif