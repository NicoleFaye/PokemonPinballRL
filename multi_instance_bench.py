#!/usr/bin/env python3
"""
Test whether multiple training instances (4x8 envs) outperform single instance (1x32 envs)
"""

import os
import time
import subprocess
import multiprocessing as mp
import psutil
import json
import argparse
from pathlib import Path
from datetime import datetime
import signal
import sys

class MultiInstanceBenchmark:
    def __init__(self, output_dir="multi_instance_benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.processes = []
        
    def run_single_instance(self, num_envs, timesteps, port_offset=0):
        """Run a single training instance"""
        # Create unique session ID
        session_id = f"bench_{num_envs}envs_{port_offset}_{int(time.time())}"
        
        cmd = [
            "python", "train.py",
            "--timesteps", str(timesteps),
            "--num-cpu", str(num_envs),
            "--headless",
            "--no-wandb",
            "--visual-mode", "game_area",  # Fastest mode
            "--frame-stack", "3",
            "--n-steps", "1024",
            "--batch-size", "256",
            "--reward-mode", "basic",
            "--info-level", "1",
            "--max-episode-frames", "5000",
            "--no-frame-stack-extra-observation",
            "--session-id", session_id
        ]
        
        # Set CPU affinity for each instance
        if port_offset > 0:
            # Calculate CPU cores for this instance
            cores_per_instance = psutil.cpu_count() // 4
            start_core = port_offset * cores_per_instance
            end_core = min(start_core + cores_per_instance - 1, psutil.cpu_count() - 1)
            
            # Use taskset to pin to specific cores
            cmd = ["taskset", f"-c", f"{start_core}-{end_core}"] + cmd
        
        print(f"Starting instance {port_offset} with {num_envs} envs on cores {start_core if port_offset > 0 else 'all'}-{end_core if port_offset > 0 else 'all'}")
        
        start_time = time.time()
        process = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        
        return {
            'process': process,
            'start_time': start_time,
            'session_id': session_id,
            'num_envs': num_envs,
            'port_offset': port_offset
        }
    
    def monitor_processes(self, process_infos, duration):
        """Monitor multiple processes and collect performance metrics"""
        metrics = {info['session_id']: [] for info in process_infos}
        
        start_time = time.time()
        while time.time() - start_time < duration:
            for info in process_infos:
                proc = info['process']
                if proc.poll() is None:  # Process still running
                    try:
                        # Get process CPU and memory usage
                        ps_proc = psutil.Process(proc.pid)
                        cpu_percent = ps_proc.cpu_percent()
                        memory_mb = ps_proc.memory_info().rss / 1024 / 1024
                        
                        metrics[info['session_id']].append({
                            'timestamp': time.time() - start_time,
                            'cpu_percent': cpu_percent,
                            'memory_mb': memory_mb
                        })
                    except psutil.NoSuchProcess:
                        pass
            
            time.sleep(1)
        
        return metrics
    
    def test_configurations(self, timesteps=50000, test_duration=120):
        """Test different configurations"""
        
        configurations = [
            {
                'name': 'single_32_envs',
                'description': '1 instance × 32 environments',
                'instances': [{'num_envs': 32, 'port_offset': 0}]
            },
            {
                'name': 'quad_8_envs', 
                'description': '4 instances × 8 environments each',
                'instances': [
                    {'num_envs': 8, 'port_offset': 0},
                    {'num_envs': 8, 'port_offset': 1}, 
                    {'num_envs': 8, 'port_offset': 2},
                    {'num_envs': 8, 'port_offset': 3}
                ]
            },
            {
                'name': 'dual_16_envs',
                'description': '2 instances × 16 environments each', 
                'instances': [
                    {'num_envs': 16, 'port_offset': 0},
                    {'num_envs': 16, 'port_offset': 1}
                ]
            }
        ]
        
        results = {}
        
        for config in configurations:
            print(f"\n{'='*60}")
            print(f"Testing: {config['description']}")
            print(f"{'='*60}")
            
            # Start all instances for this configuration
            process_infos = []
            for instance_config in config['instances']:
                info = self.run_single_instance(
                    instance_config['num_envs'], 
                    timesteps,
                    instance_config['port_offset']
                )
                process_infos.append(info)
                time.sleep(2)  # Brief delay between starts
            
            # Monitor for specified duration
            print(f"Monitoring {len(process_infos)} processes for {test_duration} seconds...")
            metrics = self.monitor_processes(process_infos, test_duration)
            
            # Terminate all processes
            for info in process_infos:
                if info['process'].poll() is None:
                    info['process'].terminate()
                    info['process'].wait(timeout=10)
            
            # Calculate results
            total_envs = sum(info['num_envs'] for info in process_infos)
            
            # Estimate FPS from monitoring
            # This is approximate - in real scenario you'd parse training logs
            avg_cpu_usage = []
            total_memory = 0
            
            for session_metrics in metrics.values():
                if session_metrics:
                    avg_cpu = sum(m['cpu_percent'] for m in session_metrics) / len(session_metrics)
                    avg_cpu_usage.append(avg_cpu)
                    total_memory += max(m['memory_mb'] for m in session_metrics)
            
            results[config['name']] = {
                'description': config['description'],
                'total_environments': total_envs,
                'num_instances': len(process_infos),
                'avg_cpu_usage': sum(avg_cpu_usage) / len(avg_cpu_usage) if avg_cpu_usage else 0,
                'total_memory_mb': total_memory,
                'estimated_efficiency': sum(avg_cpu_usage) / (psutil.cpu_count() * 100) if avg_cpu_usage else 0,
                'metrics': metrics
            }
            
            print(f"Results for {config['name']}:")
            print(f"  Total environments: {total_envs}")
            print(f"  Average CPU usage: {results[config['name']]['avg_cpu_usage']:.1f}%")
            print(f"  Total memory: {total_memory:.0f} MB")
            print(f"  CPU efficiency: {results[config['name']]['estimated_efficiency']*100:.1f}%")
        
        return results
    
    def run_actual_fps_test(self, timesteps=25000):
        """Run actual FPS test by parsing training output"""
        
        configs_to_test = [
            {'name': 'single_32', 'instances': 1, 'envs_per': 32},
            {'name': 'quad_8', 'instances': 4, 'envs_per': 8},
            {'name': 'dual_16', 'instances': 2, 'envs_per': 16},
        ]
        
        results = {}
        
        for config in configs_to_test:
            print(f"\n{'='*50}")
            print(f"FPS Test: {config['instances']} instances × {config['envs_per']} envs")
            print(f"{'='*50}")
            
            if config['instances'] == 1:
                # Single instance
                cmd = [
                    "python", "bench.py",
                    "--timesteps", str(timesteps),
                    "--iterations", "1"
                ]
                
                # Modify bench.py env_counts for this test
                env_counts = [config['envs_per']]
                
            else:
                # Multiple instances - we'll need to run them in parallel
                # and aggregate results
                print(f"Running {config['instances']} parallel instances...")
                
                # This would require modifying bench.py or running separate processes
                # For now, let's create a simplified version
                fps_results = []
                
                for i in range(config['instances']):
                    print(f"  Starting instance {i+1}/{config['instances']}")
                    
                    cmd = [
                        "timeout", "60",  # 60 second test
                        "python", "test.py"  # Your simple test script
                    ]
                    
                    # Run with CPU affinity
                    cores_per_instance = psutil.cpu_count() // config['instances']
                    start_core = i * cores_per_instance
                    end_core = min(start_core + cores_per_instance - 1, psutil.cpu_count() - 1)
                    
                    cmd = ["taskset", f"-c", f"{start_core}-{end_core}"] + cmd
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=70)
                        # Parse FPS from output (you'd need to modify test.py to output FPS)
                        # fps = parse_fps_from_output(result.stdout)
                        # fps_results.append(fps)
                    except subprocess.TimeoutExpired:
                        print(f"    Instance {i+1} timed out")
                
                # Aggregate FPS results
                # total_fps = sum(fps_results)
                # results[config['name']] = total_fps
            
        return results

def create_simple_fps_test():
    """Create a simple FPS measurement script"""
    script = '''#!/usr/bin/env python3
"""
Simple FPS test for Pokemon Pinball environment
"""

import time
import argparse
from pokemon_pinball_env import PokemonPinballEnv, DEFAULT_CONFIG
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

def make_env(rank, config, seed=0):
    def _init():
        env = PokemonPinballEnv("./roms/pokemon_pinball.gbc", config)
        env = Monitor(env)
        env.reset(seed=(seed + rank))
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--steps', type=int, default=10000)
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config['headless'] = True
    config['visual_mode'] = 'game_area'
    config['frame_stack'] = 3
    config['frame_stack_extra_observation'] = False
    
    # Create environments
    env = SubprocVecEnv([make_env(i, config) for i in range(args.envs)])
    
    # Run test
    obs = env.reset()
    start_time = time.time()
    
    for step in range(args.steps):
        actions = [env.action_space.sample() for _ in range(args.envs)]
        obs, rewards, dones, infos = env.step(actions)
        
        if step % 1000 == 0:
            elapsed = time.time() - start_time
            fps = (step + 1) * args.envs / elapsed
            print(f"Step {step}: {fps:.1f} FPS")
    
    total_time = time.time() - start_time
    total_fps = args.steps * args.envs / total_time
    
    print(f"Final: {total_fps:.1f} FPS with {args.envs} environments")
    env.close()

if __name__ == "__main__":
    main()
'''
    
    with open("simple_fps_test.py", "w") as f:
        f.write(script)
    
    return "simple_fps_test.py"

def main():
    parser = argparse.ArgumentParser(description='Multi-instance vs single-instance benchmark')
    parser.add_argument('--timesteps', type=int, default=25000, help='Timesteps per test')
    parser.add_argument('--test-duration', type=int, default=120, help='Test duration in seconds')
    parser.add_argument('--mode', choices=['monitor', 'fps'], default='fps', help='Test mode')
    args = parser.parse_args()
    
    # Create simple FPS test script
    fps_script = create_simple_fps_test()
    print(f"Created {fps_script} for FPS testing")
    
    benchmark = MultiInstanceBenchmark()
    
    if args.mode == 'monitor':
        results = benchmark.test_configurations(args.timesteps, args.test_duration)
    else:
        # Run simplified FPS tests
        print("Running FPS comparison tests...")
        
        configs = [
            {'envs': 32, 'name': 'single_32'},
            {'envs': 8, 'name': 'single_8_baseline'},
        ]
        
        for config in configs:
            print(f"\nTesting {config['envs']} environments...")
            cmd = ["python", fps_script, "--envs", str(config['envs']), "--steps", "5000"]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                print(f"Output for {config['name']}:")
                print(result.stdout)
                if result.stderr:
                    print(f"Errors: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"Test timed out for {config['name']}")
    
    print("\nRecommendations:")
    print("1. Run the simple FPS test with different configurations")
    print("2. Monitor system resources with htop during each test")
    print("3. Compare CPU efficiency and memory usage")
    print("4. Test with CPU affinity (taskset)")

if __name__ == "__main__":
    main()