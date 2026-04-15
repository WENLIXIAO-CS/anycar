export const ACTION_LABELS: Record<string, string[]> = {
  holonomic: ['v_fwd', 'v_side', 'omega'],
  robocasa_holonomic: ['v_fwd', 'v_side', 'omega'],
  unicycle: ['accel', 'steer'],
  diff_drive: ['v_left', 'v_right'],
};

export const ACTION_COLORS = ['#f59e0b', '#10b981', '#8b5cf6', '#ef4444', '#06b6d4'];
