#include <pybind11/embed.h>
namespace py = pybind11;

int main()
{
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    py::object sys = py::module::import("sys");
    py::print("Python version", sys.attr("version"));

    py::object torch = py::module::import("torch");
    py::print("Torch version", torch.attr("__version__"));

    return 0;
}
