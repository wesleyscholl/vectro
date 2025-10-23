#!/usr/bin/env python3
"""
Visual Benchmarking for Vectro Backends

Shows animated real-time performance comparisons between:
- Mojo (if available)
- Cython (if available)
- NumPy (fallback)
- PQ (memory-efficient)

Usage:
    python visual_bench.py --n 5000 --d 128 --duration 30
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from python import interface
try:
    from python import mojo_runner
except Exception:
    mojo_runner = None


class VisualBenchmark:
    """Real-time visual benchmarking with animated charts."""

    def __init__(self, n_vectors=1000, dimensions=128, duration=30):
        self.n = n_vectors
        self.d = dimensions
        self.duration = duration
        self.start_time = time.time()

        # Generate test data
        rng = np.random.default_rng(42)
        self.embeddings = rng.standard_normal((self.n, self.d)).astype(np.float32)

        # Backend detection
        # Detect backends. Mojo can be available either as a runtime (mojo CLI)
        # or as a compiled Python module; we consider it available if either
        # the Python shim is present or the mojo CLI can run.
        mojo_runtime_ok = mojo_runner is not None and mojo_runner.mojo_available()
        self.backends = {
            'mojo': (interface._mojo_quant is not None) or mojo_runtime_ok,
            'cython': interface._cython_quant is not None,
            'numpy': True,  # Always available
        }

        # Performance tracking
        self.performance_data = {
            'mojo': {'throughput': [], 'quality': [], 'times': []},
            'cython': {'throughput': [], 'quality': [], 'times': []},
            'numpy': {'throughput': [], 'quality': [], 'times': []},
        }

        # Setup matplotlib
        plt.style.use('dark_background')
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Vectro Backend Performance Comparison', fontsize=16, color='white')

        # Initialize plots
        self.lines = {}
        self.bars = {}
        self.setup_plots()

    def setup_plots(self):
        """Initialize the four subplots."""
        # Throughput over time
        self.ax1.set_title('Quantization Throughput (vec/s)', color='white')
        self.ax1.set_xlabel('Time (s)', color='white')
        self.ax1.set_ylabel('Throughput', color='white')
        self.ax1.grid(True, alpha=0.3)

        # Quality comparison
        self.ax2.set_title('Reconstruction Quality (Cosine Similarity)', color='white')
        self.ax2.set_xlabel('Time (s)', color='white')
        self.ax2.set_ylabel('Mean Cosine Similarity', color='white')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0.995, 1.001)

        # Memory usage
        self.ax3.set_title('Memory Efficiency', color='white')
        self.ax3.set_xlabel('Backend', color='white')
        self.ax3.set_ylabel('Compression Ratio', color='white')
        self.ax3.grid(True, alpha=0.3)

        # Real-time status
        self.ax4.set_title('Benchmark Status', color='white')
        self.ax4.axis('off')

        # Initialize line plots
        colors = {'mojo': '#FF6B6B', 'cython': '#4ECDC4', 'numpy': '#45B7D1'}
        for backend in self.backends:
            if self.backends[backend]:
                line1, = self.ax1.plot([], [], color=colors[backend], label=backend.upper(), linewidth=2)
                line2, = self.ax2.plot([], [], color=colors[backend], label=backend.upper(), linewidth=2)
                self.lines[backend] = {'throughput': line1, 'quality': line2}

        self.ax1.legend()
        self.ax2.legend()

    def benchmark_backend(self, backend_name, force_backend=None):
        """Run a single benchmark iteration for a backend."""
        try:
            # Force specific backend if requested
            if force_backend == 'numpy':
                # Temporarily disable other backends
                saved_mojo = interface._mojo_quant
                saved_cython = interface._cython_quant
                interface._mojo_quant = None
                interface._cython_quant = None

            start_time = time.time()

            # Quantize
            quant_start = time.time()
            compressed = interface.quantize_embeddings(self.embeddings)
            quant_time = time.time() - quant_start

            # Reconstruct
            recon_start = time.time()
            reconstructed = interface.reconstruct_embeddings(
                compressed['q'], compressed['scales'], compressed['dims']
            )
            recon_time = time.time() - recon_start

            total_time = time.time() - start_time

            # Calculate metrics
            throughput = self.n / quant_time if quant_time > 0 else 0
            quality = interface.mean_cosine_similarity(self.embeddings, reconstructed)

            # Calculate compression ratio
            orig_bytes = self.embeddings.nbytes
            comp_bytes = compressed['q'].nbytes + compressed['scales'].nbytes
            compression_ratio = comp_bytes / orig_bytes

            # Restore backends
            if force_backend == 'numpy':
                interface._mojo_quant = saved_mojo
                interface._cython_quant = saved_cython

            return {
                'throughput': throughput,
                'quality': quality,
                'compression_ratio': compression_ratio,
                'quant_time': quant_time,
                'recon_time': recon_time,
                'total_time': total_time
            }

        except Exception as e:
            print(f"Error benchmarking {backend_name}: {e}")
            return None

    def update_frame(self, frame):
        """Update the animation frame."""
        current_time = time.time() - self.start_time

        # Run benchmarks for each available backend
        for backend in self.backends:
            if self.backends[backend]:
                # If mojo CLI is available but not imported as a module, run the
                # external mojo test runner to get its metrics (faster than
                # attempting to compile to a Python extension here).
                if backend == 'mojo' and mojo_runner is not None and interface._mojo_quant is None:
                    # Run external mojo test and use its reported throughput
                    mj = mojo_runner.run_mojo_benchmark()
                    if mj is not None:
                        result = {
                            'throughput': mj['throughput'],
                            'quality': mj['quality'],
                            'compression_ratio': 0.25,  # placeholder
                            'quant_time': mj.get('quant_time', 0.0),
                            'recon_time': mj.get('recon_time', 0.0),
                            'total_time': mj.get('quant_time', 0.0) + mj.get('recon_time', 0.0)
                        }
                    else:
                        result = None
                elif backend == 'numpy':
                    result = self.benchmark_backend(backend, force_backend='numpy')
                else:
                    result = self.benchmark_backend(backend)

                if result:
                    self.performance_data[backend]['throughput'].append(result['throughput'])
                    self.performance_data[backend]['quality'].append(result['quality'])
                    self.performance_data[backend]['times'].append(current_time)

                    # Update line plots
                    times = self.performance_data[backend]['times']
                    throughput = self.performance_data[backend]['throughput']
                    quality = self.performance_data[backend]['quality']

                    self.lines[backend]['throughput'].set_data(times, throughput)
                    self.lines[backend]['quality'].set_data(times, quality)

        # Update axis limits
        if self.performance_data['numpy']['times']:
            max_time = max(max(data['times']) for data in self.performance_data.values() if data['times'])
            self.ax1.set_xlim(0, max(max_time, 1))
            self.ax2.set_xlim(0, max(max_time, 1))

            # Update throughput axis
            all_throughput = []
            for data in self.performance_data.values():
                all_throughput.extend(data['throughput'])
            if all_throughput:
                self.ax1.set_ylim(0, max(all_throughput) * 1.1)

        # Update compression ratio bars
        self.ax3.clear()
        self.ax3.set_title('Memory Efficiency', color='white')
        self.ax3.set_xlabel('Backend', color='white')
        self.ax3.set_ylabel('Compression Ratio (lower is better)', color='white')
        self.ax3.grid(True, alpha=0.3)

        backends = []
        ratios = []
        colors = []
        color_map = {'mojo': '#FF6B6B', 'cython': '#4ECDC4', 'numpy': '#45B7D1'}

        for backend in self.backends:
            if self.backends[backend] and self.performance_data[backend]['throughput']:
                # Get latest compression ratio
                latest_result = self.benchmark_backend(backend)
                if latest_result:
                    backends.append(backend.upper())
                    ratios.append(latest_result['compression_ratio'])
                    colors.append(color_map[backend])

        if ratios:
            bars = self.ax3.bar(backends, ratios, color=colors, alpha=0.7)
            self.ax3.set_ylim(0, max(ratios) * 1.2)

        # Update status display
        self.ax4.clear()
        self.ax4.set_title('Benchmark Status', color='white')
        self.ax4.axis('off')

        status_text = f"Time: {current_time:.1f}s\n"
        for backend in ['mojo', 'cython', 'numpy']:
            if self.backends.get(backend, False):
                count = len(self.performance_data[backend]['throughput'])
                if count > 0:
                    latest = self.performance_data[backend]['throughput'][-1]
                    status_text += f"{backend.upper()}: {latest:.0f} vec/s\n"
                else:
                    status_text += f"{backend.upper()}: Initializing...\n"
            else:
                if backend == 'mojo':
                    status_text += f"{backend.upper()}: Not Installed\n"
                else:
                    status_text += f"{backend.upper()}: Not Available\n"

        self.ax4.text(0.1, 0.8, status_text, transform=self.ax4.transAxes,
                     fontsize=10, verticalalignment='top', color='white',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

        return self.lines

    def run_animation(self):
        """Run the animated benchmark."""
        print(f"Starting visual benchmark with {self.n} vectors of {self.d} dimensions")
        print(f"Duration: {self.duration} seconds")
        print(f"Available backends: {[b for b, avail in self.backends.items() if avail]}")
        if not self.backends['mojo']:
            print("Note: Mojo backend not available. Install Modular CLI to enable Mojo performance testing.")

        ani = animation.FuncAnimation(
            self.fig, self.update_frame, frames=self.duration * 2,  # 2 FPS
            interval=500, blit=False, repeat=False
        )

        plt.tight_layout()
        plt.show()

        # Print final results
        print("\n" + "="*60)
        print("FINAL BENCHMARK RESULTS")
        print("="*60)

        for backend in ['mojo', 'cython', 'numpy']:
            if self.backends.get(backend, False) and self.performance_data[backend]['throughput']:
                throughput = self.performance_data[backend]['throughput']
                quality = self.performance_data[backend]['quality']

                avg_throughput = np.mean(throughput)
                avg_quality = np.mean(quality)
                best_throughput = max(throughput)

                print(f"\n{backend.upper()}:")
                print(f"  Avg Throughput: {avg_throughput:.0f} vec/s")
                print(f"  Avg Quality: {avg_quality:.4f}")
                print(f"  Best Throughput: {best_throughput:.0f} vec/s")
                print(f"  Samples: {len(throughput)}")


def main():
    parser = argparse.ArgumentParser(description='Visual benchmarking for Vectro backends')
    parser.add_argument('--n', type=int, default=1000, help='Number of vectors to benchmark')
    parser.add_argument('--d', type=int, default=128, help='Vector dimensions')
    parser.add_argument('--duration', type=int, default=30, help='Benchmark duration in seconds')
    args = parser.parse_args()

    benchmark = VisualBenchmark(args.n, args.d, args.duration)
    benchmark.run_animation()


if __name__ == '__main__':
    main()