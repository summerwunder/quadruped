import numpy as np 


class SwingTrajectoryGenerator:
    def __init__(self, swing_height: float, swing_duration: float):
        self.swing_height = swing_height
        self.swing_duration = swing_duration
        self.half_swing_duration = swing_duration / 2.0
        self.factor_duration = 1.0 / self.half_swing_duration   # [0 , 1]
    
    def _evaluate_bezier(self, t_rel:float, p1, p2, p3, p4):
        t = t_rel
        t_inv = 1 - t
        pos = (t_inv**3) * p1 + \
              3 * (t_inv**2) * t * p2 + \
              3 * t_inv * (t**2) * p3 + \
              (t**3) * p4
        vel = 3 * ( (t_inv**2) * (p2 - p1) + \
                    2 * t_inv * t * (p3 - p2) + \
                    (t**2) * (p4 - p3) ) * self.factor_duration
        acc = 6 * ( (t_inv) * (p3 - 2 * p2 + p1) + \
                    t * (p4 - 2 * p3 + p2) ) * (self.factor_duration ** 2)
        return pos.flatten(), vel.flatten(), acc.flatten()

    def get_swing_reference_trajectory(self, 
                                       t:float, 
                                       lift_off:np.ndarray, 
                                       touch_down:np.ndarray)-> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate swing leg reference trajectory
        Args:
            t: current swing time (s)
            lift_off: lift-off position (3,)
            touch_down: touch-down position (3,)
        Returns:
            des_foot_pos: desired foot position (3,)
            des_foot_vel: desired foot velocity (3,)
            des_foot_acc: desired foot acceleration (3,)
        """
        
        t = np.clip(t, 0.0, self.swing_duration)
        lift_off = lift_off.reshape(3)
        touch_down = touch_down.reshape(3)
        mid_point = 0.5 * (lift_off + touch_down)

        if t <= self.half_swing_duration:
            t_rel = t * self.factor_duration
            p1 = lift_off
            p2 = lift_off
            p3 = np.array([lift_off[0],lift_off[1], self.swing_height])
            p4 = np.array([mid_point[0], mid_point[1], self.swing_height])
            return self._evaluate_bezier(t_rel, p1, p2, p3, p4)
        else:
            t_rel = (t - self.half_swing_duration) * self.factor_duration
            p1 = np.array([mid_point[0], mid_point[1], self.swing_height])
            p2 = np.array([touch_down[0], touch_down[1], self.swing_height])
            p3 = touch_down
            p4 = touch_down
            return self._evaluate_bezier(t_rel, p1, p2, p3, p4)
    
    def plot_trajectory(self, lift_off: np.ndarray, touch_down: np.ndarray, 
                       num_samples: int = 100, save_path: str = None):
        """Plot complete swing trajectory
        
        Args:
            lift_off: lift-off position (3,)
            touch_down: touch-down position (3,)
            num_samples: number of samples
            save_path: save path, if None then show plot
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        t_array = np.linspace(0, self.swing_duration, num_samples)
        
        positions = []
        velocities = []
        accelerations = []
        
        for t in t_array:
            pos, vel, acc = self.get_swing_reference_trajectory(t, lift_off, touch_down)
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)
        
        positions = np.array(positions)  # (N, 3)
        velocities = np.array(velocities)  # (N, 3)
        accelerations = np.array(accelerations)  # (N, 3)
        
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 3D trajectory
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
        ax1.scatter(*lift_off, color='green', s=100, marker='o', label='Lift-off')
        ax1.scatter(*touch_down, color='red', s=100, marker='s', label='Touch-down')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Swing Trajectory')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Position vs Time
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(t_array, positions[:, 0], 'r-', label='X')
        ax2.plot(t_array, positions[:, 1], 'g-', label='Y')
        ax2.plot(t_array, positions[:, 2], 'b-', label='Z')
        ax2.axhline(y=self.swing_height, color='k', linestyle='--', alpha=0.3, label='Swing height')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.set_title('Position vs Time')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Velocity vs Time
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(t_array, velocities[:, 0], 'r-', label='Vx')
        ax3.plot(t_array, velocities[:, 1], 'g-', label='Vy')
        ax3.plot(t_array, velocities[:, 2], 'b-', label='Vz')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_title('Velocity vs Time')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Acceleration vs Time
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(t_array, accelerations[:, 0], 'r-', label='Ax')
        ax4.plot(t_array, accelerations[:, 1], 'g-', label='Ay')
        ax4.plot(t_array, accelerations[:, 2], 'b-', label='Az')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Acceleration (m/s²)')
        ax4.set_title('Acceleration vs Time')
        ax4.legend()
        ax4.grid(True)
        
        # 5. XY plane projection
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
        ax5.scatter(lift_off[0], lift_off[1], color='green', s=100, marker='o', label='Lift-off')
        ax5.scatter(touch_down[0], touch_down[1], color='red', s=100, marker='s', label='Touch-down')
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        ax5.set_title('XY Plane Projection')
        ax5.legend()
        ax5.grid(True)
        ax5.axis('equal')
        
        # 6. Velocity and acceleration magnitude
        ax6 = fig.add_subplot(2, 3, 6)
        vel_magnitude = np.linalg.norm(velocities, axis=1)
        acc_magnitude = np.linalg.norm(accelerations, axis=1)
        ax6.plot(t_array, vel_magnitude, 'b-', label='Velocity magnitude')
        ax6_twin = ax6.twinx()
        ax6_twin.plot(t_array, acc_magnitude, 'r-', label='Acceleration magnitude')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Velocity magnitude (m/s)', color='b')
        ax6_twin.set_ylabel('Acceleration magnitude (m/s²)', color='r')
        ax6.set_title('Velocity and Acceleration Magnitude')
        ax6.tick_params(axis='y', labelcolor='b')
        ax6_twin.tick_params(axis='y', labelcolor='r')
        ax6.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
        
        return fig


if __name__ == '__main__':
    print("=== Swing Trajectory Generator Test ===\n")
    
    # 1. Create generator
    swing_height = 0.08  # 8cm swing height
    swing_duration = 0.3  # 300ms swing duration
    generator = SwingTrajectoryGenerator(swing_height, swing_duration)
    print(f"Created generator: swing_height={swing_height}m, swing_duration={swing_duration}s\n")
    
    # 2. Define lift-off and touch-down positions
    lift_off = np.array([0.0, 0.1, 0.0])  # start point
    touch_down = np.array([0.2, 0.1, 0.0])  # end point (0.2m forward)
    print(f"Lift-off position: {lift_off}")
    print(f"Touch-down position: {touch_down}\n")
    
    # 3. Test trajectory at key moments
    print("=== Key Moments Trajectory Test ===\n")
    test_times = [0.0, swing_duration/4, swing_duration/2, 3*swing_duration/4, swing_duration]
    
    for t in test_times:
        pos, vel, acc = generator.get_swing_reference_trajectory(t, lift_off, touch_down)
        print(f"t = {t:.3f}s:")
        print(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        print(f"  Velocity: [{vel[0]:.4f}, {vel[1]:.4f}, {vel[2]:.4f}]")
        print(f"  Acceleration: [{acc[0]:.4f}, {acc[1]:.4f}, {acc[2]:.4f}]")
        print()
    
    # 4. Verify boundary conditions
    print("=== Boundary Conditions Verification ===\n")
    
    # Start point verification
    pos_start, vel_start, _ = generator.get_swing_reference_trajectory(0.0, lift_off, touch_down)
    pos_error_start = np.linalg.norm(pos_start - lift_off)
    print(f"Start position error: {pos_error_start:.6f} m")
    print(f"Start velocity: {vel_start} m/s")
    
    # End point verification
    pos_end, vel_end, _ = generator.get_swing_reference_trajectory(swing_duration, lift_off, touch_down)
    pos_error_end = np.linalg.norm(pos_end - touch_down)
    print(f"End position error: {pos_error_end:.6f} m")
    print(f"End velocity: {vel_end} m/s")
    
    # Maximum height verification
    t_samples = np.linspace(0, swing_duration, 100)
    max_height = 0.0
    for t in t_samples:
        pos, _, _ = generator.get_swing_reference_trajectory(t, lift_off, touch_down)
        max_height = max(max_height, pos[2])
    print(f"Actual max height: {max_height:.4f} m (expected: {swing_height:.4f} m)")
    
    # Verification results
    tolerance = 1e-6
    if pos_error_start < tolerance and pos_error_end < tolerance:
        print("\nBoundary conditions verification PASSED")
    else:
        print(f"\nBoundary conditions verification FAILED! Start error: {pos_error_start}, End error: {pos_error_end}")
    
    # 5. Plot trajectory
    print("\n=== Plot Trajectory ===\n")
    try:
        generator.plot_trajectory(lift_off, touch_down, num_samples=100)
        print("Trajectory plot completed")
    except Exception as e:
        print(f"Trajectory plot failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test Completed ===")
