"""
Test Coverage Reporter for Vectro
Tracks which functions/features are tested across all modules
"""

struct TestCoverage:
    """Track test coverage statistics."""
    var total_functions: Int
    var tested_functions: Int
    var total_lines: Int
    var tested_lines: Int
    var module_name: String
    
    fn __init__(inout self, name: String):
        self.module_name = name
        self.total_functions = 0
        self.tested_functions = 0
        self.total_lines = 0
        self.tested_lines = 0
    
    fn add_function(inout self, tested: Bool = True):
        """Record a function."""
        self.total_functions += 1
        if tested:
            self.tested_functions += 1
    
    fn add_lines(inout self, total: Int, tested: Int):
        """Record line coverage."""
        self.total_lines += total
        self.tested_lines += tested
    
    fn get_function_coverage(self) -> Float64:
        """Get function coverage percentage."""
        if self.total_functions == 0:
            return 0.0
        return 100.0 * Float64(self.tested_functions) / Float64(self.total_functions)
    
    fn get_line_coverage(self) -> Float64:
        """Get line coverage percentage."""
        if self.total_lines == 0:
            return 0.0
        return 100.0 * Float64(self.tested_lines) / Float64(self.total_lines)
    
    fn print_report(self):
        """Print coverage report."""
        print("  Module:", self.module_name)
        print("    Functions:", self.tested_functions, "/", self.total_functions, 
              "(", Int(self.get_function_coverage()), "%)")
        print("    Lines:", self.tested_lines, "/", self.total_lines,
              "(", Int(self.get_line_coverage()), "%)")


struct CoverageTracker:
    """Global coverage tracker."""
    var modules: List[TestCoverage]
    
    fn __init__(inout self):
        self.modules = List[TestCoverage]()
    
    fn add_module(inout self, coverage: TestCoverage):
        """Add a module's coverage."""
        self.modules.append(coverage)
    
    fn print_summary(self):
        """Print overall coverage summary."""
        print("\n" + "=" * 70)
        print("TEST COVERAGE REPORT")
        print("=" * 70)
        
        var total_funcs = 0
        var tested_funcs = 0
        var total_lines = 0
        var tested_lines = 0
        
        for i in range(len(self.modules)):
            var mod = self.modules[i]
            mod.print_report()
            total_funcs += mod.total_functions
            tested_funcs += mod.tested_functions
            total_lines += mod.total_lines
            tested_lines += mod.tested_lines
        
        print("\n" + "=" * 70)
        print("OVERALL COVERAGE")
        print("=" * 70)
        
        var func_cov = 100.0 * Float64(tested_funcs) / Float64(total_funcs) if total_funcs > 0 else 0.0
        var line_cov = 100.0 * Float64(tested_lines) / Float64(total_lines) if total_lines > 0 else 0.0
        
        print("  Functions:", tested_funcs, "/", total_funcs, "(", Int(func_cov), "%)")
        print("  Lines:", tested_lines, "/", total_lines, "(", Int(line_cov), "%)")
        
        print("\n" + "=" * 70)
        if func_cov >= 80.0 and line_cov >= 80.0:
            print("✅ COVERAGE TARGET MET (≥80%)")
        else:
            print("⚠️  COVERAGE BELOW TARGET (<80%)")
            print("   Target: 80% function and line coverage")
            print("   Function coverage:", Int(func_cov), "% (need", 80 - Int(func_cov), "% more)")
            print("   Line coverage:", Int(line_cov), "% (need", 80 - Int(line_cov), "% more)")
        print("=" * 70)
