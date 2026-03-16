## 1. What Does This Code Do?

When a wind turbine operates, its blades `can bend (deflect) under heavy wind loads. If a blade bends too much, it risks hitting the tower.

This script is a **computer vision and data analysis pipeline** designed to detect and measure that bending in real-time. It operates by combining two data streams:

1. **Video Footage (`clockwise_8rpm.mp4`)**: An overhead or front-view camera recording the spinning turbine and that data will be send thru UDP to communicate each other but it is next step.
2. **Telemetry Data (`cc_8rpm_analyzed_data.csv`)**: A precise log of the exact angle of every blade at every frame of the video.

The code sits and watches the telemetry data. **When it sees a blade approach a perfectly parallel (horizontal) position (Angle ~ 0 degrees), it triggers an image analysis routine.** It isolates the blade from the background, extracts its central spine (centerline), and mathematically measures how much the outer half of the blade (midpoint to tip) bends away from a perfectly straight line. As with the 8rpm clockwise video, results are accurate and further testing with different speed is under way.

## 2. Geometry:

Once `PEAK_SCAN` locks onto the perfect frame, [get_blade_centerline_analysis] takes over. This is where the heavy mathematics happens.

1. **Mask Generation:** We use `MOG2` (Background Subtraction) to separate the moving white blades from the static background. The code aggressively filters out shadows (values = `127`) and noise using Gaussian blur and Morphological Operations (Erosion/Dilation).
2. **Finding the Extents (Dot Product Projection):** We draw an imaginary vector line (`vx, vy`) straight through the middle of the blade. We then use **Dot Product Scalar Projection** to smash every single white pixel onto that imaginary line. The pixel that lands furthest back is the **Root (P1)**. The pixel that lands furthest forward is the **Tip (P3)**.
3. **Centerline Trace (Rotational Scanning & Noise Rejection):** The computer rotates the image entirely so the blade is perfectly flat on the screen. It then scans through the blade from left-to-right, looking at one vertical column of pixels at a time.

- _Note:_ The code also deliberately ignores the first 15% of the blade nearest to the hub, because the hub block is thick and asymmetrical, which would falsely distort the centerline trace.

4. **Calculus-Inspired Smoothing:** The raw centerline is jagged due to pixelation. A discrete convolution (`np.convolve`) is run over the points. This is effectively a running average that mathematical smoothens the jagged pixels into a fluid, accurate curve.
5. **The Deflection Math:**
   We measure bending exclusively between the Midpoint (P2) and the Tip (P3).
   - First, we calculate exactly where the midpoint is by checking which point lies physically halfway along the vector projection span.
   - Then, we create a mathematical Straight Line connecting **P2** and **P3**.
   - The single point with the largest distance is flagged as the **Max Deflection**. If this distance exceeds `DEFLECTION_THRESHOLD` (10 pixels), the system alerts that the blade is bending dangerously.

## 3. Code Explanation:
