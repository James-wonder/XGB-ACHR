import cv2
import numpy as np

def extract_roc_coordinates(image_path):
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print("错误：无法找到图片，请检查路径。")
        return

    # 转换颜色空间 BGR -> HSV (方便提取颜色)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # -------------------------------------------------
    # 2. 定位图表区域 (寻找黑色的坐标轴框)
    # -------------------------------------------------
    # 将图片转灰度并二值化，寻找黑色矩形框
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plot_areas = []
    
    # 筛选出符合图表形状的矩形框
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 过滤掉太小的噪点或太大的整体边框 (根据实际图片分辨率调整这些阈值)
        if w > 100 and h > 100 and w < img.shape[1] - 10:
            # 这是一个潜在的图表区域
            plot_areas.append((x, y, w, h))

    # 从左上到右下排序图表，保证输出顺序一致
    # 简单的按 y 排序，再按 x 排序 (分行处理)
    plot_areas.sort(key=lambda b: (b[1] // 100, b[0]))

    print(f"检测到 {len(plot_areas)} 个图表区域。\n")

    # -------------------------------------------------
    # 3. 遍历每个图表，提取红色曲线
    # -------------------------------------------------
    for i, (px, py, pw, ph) in enumerate(plot_areas):
        # 裁剪出当前的子图
        roi = img[py:py+ph, px:px+pw]
        roi_hsv = hsv[py:py+ph, px:px+pw]

        # 定义红色的HSV范围 (根据图片中的淡红色调整)
        # 红色在HSV中跨越 0 和 180，所以需要两个mask
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(roi_hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(roi_hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2

        # -------------------------------------------------
        # 4. 提取曲线骨架并简化坐标
        # -------------------------------------------------
        # 寻找红色线条的轮廓
        line_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not line_contours:
            print(f"图表 {i+1}: 未检测到红色曲线")
            continue

        # 假设最大的轮廓就是ROC曲线
        c = max(line_contours, key=cv2.contourArea)

        # 使用 approxPolyDP 算法简化曲线 (提取拐点)
        # epsilon 控制简化的程度，越小越精确，越大越简化
        epsilon = 0.005 * cv2.arcLength(c, False) 
        approx = cv2.approxPolyDP(c, epsilon, False)

        # 提取坐标点
        points = []
        for point in approx:
            # 这里的坐标是相对于ROI裁剪区域的像素坐标
            px_val, py_val = point[0]
            
            # -------------------------------------------------
            # 5. 坐标归一化 (Pixel -> Data 0.0-1.0)
            # -------------------------------------------------
            # X轴: 0 在左边 -> x / width
            # Y轴: 0 在底部 -> (height - y) / height  (注意图片y轴是向下的)
            
            norm_x = px_val / pw
            norm_y = (ph - py_val) / ph
            
            # 修正边界误差 (clip)
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            
            points.append((norm_x, norm_y))

        # 按 X 轴排序，因为 approxPolyDP 出来的点可能是乱序的
        points.sort(key=lambda p: p[0])

        # -------------------------------------------------
        # 6. 输出结果
        # -------------------------------------------------
        var_name = f"roc_chart_{i+1}_data"
        print(f"{var_name} = \"\"\"")
        
        # 简单去重：如果两个点非常接近，只保留一个（可选）
        prev_x, prev_y = -1, -1
        for x, y in points:
            # 格式化字符串，保留5位小数
            print(f"{x:.5f},{y:.5f}")
        print("\"\"\"\n")

# 运行函数 (修改为你的图片文件名)
extract_roc_coordinates("no implanted/Clinical/Internal_Validation_Results_ClinicalOnly/5_ROC_Curves.png")