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

#include <iostream>
#include <cstddef>

namespace nanobind {
namespace detail {

template<typename Array, typename Entry, size_t Size>
struct glm_caster {

	NB_TYPE_CASTER(Array, io_name(NB_TYPING_SEQUENCE, NB_TYPING_LIST) +
		const_name("[") + make_caster<Entry>::Name + const_name("]"))

	using Caster = make_caster<Entry>;

	bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {

		PyObject *temp;
		/* Will initialize 'temp' (NULL in the case of a failure.) */
		PyObject **o = seq_get_with_size(src.ptr(), Size, &temp);

		Caster caster;
		bool success = o != nullptr;

		flags = flags_for_local_caster<Entry>(flags);

		if (success) {
			for (size_t i = 0; i < Size; ++i) {
				if (!caster.from_python(o[i], flags, cleanup) ||
					!caster.template can_cast<Entry>()) {
					success = false;
					break;
				}

				value[i] = caster.operator cast_t<Entry>();
			}

			Py_XDECREF(temp);
		}

		return success;
	}

	template <typename T>
	static handle from_cpp(T &&src, rv_policy policy, cleanup_list *cleanup) {
		object ret = steal(PyList_New(Size));

		if (ret.is_valid()) {
			Py_ssize_t index = 0;

			for(size_t i = 0; i < Size; ++i){
			auto& value = src[i];
				handle h = Caster::from_cpp(forward_like_<T>(value), policy, cleanup);

				if (!h.is_valid()) {
					ret.reset();
					break;
				}

				NB_LIST_SET_ITEM(ret.ptr(), index++, h.ptr());
			}
		}

		return ret.release();
	}
};

template<typename T, glm::precision P>
struct type_caster<glm::tvec2<T, P>>
 : glm_caster<glm::tvec2<T, P>, T, 2> {};

template<typename T, glm::precision P>
struct type_caster<glm::tvec3<T, P>>
 : glm_caster<glm::tvec3<T, P>, T, 3> {};

template<typename T, glm::precision P>
struct type_caster<glm::tvec4<T, P>>
 : glm_caster<glm::tvec4<T, P>, T, 4> {};

}	// end of namespace detail

}

#endif