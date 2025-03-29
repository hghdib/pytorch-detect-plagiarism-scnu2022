import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# 读取户型图
image = cv2.imread('ce.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯模糊去噪
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 轮廓检测
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建空白图像用于绘制分割结果
segmented = np.zeros_like(image)
cv2.drawContours(segmented, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# 转换为二值掩码
segmented_gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
_, pred_mask = cv2.threshold(segmented_gray, 1, 1, cv2.THRESH_BINARY)

# 读取真实标注（Ground Truth）
gt_image = cv2.imread('zhen.jpg', cv2.IMREAD_GRAYSCALE)
_, gt_mask = cv2.threshold(gt_image, 1, 1, cv2.THRESH_BINARY)


### 计算评估指标
def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union != 0 else 0


def compute_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    return (2.0 * intersection) / (pred_mask.sum() + gt_mask.sum())


def compute_precision_recall(pred_mask, gt_mask):
    TP = np.logical_and(pred_mask, gt_mask).sum()
    FP = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    FN = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    return precision, recall


# 计算Hausdorff距离
def compute_hausdorff(pred_points, gt_points):
    d1 = directed_hausdorff(pred_points, gt_points)[0]
    d2 = directed_hausdorff(gt_points, pred_points)[0]
    return max(d1, d2)


# 计算指标
iou_score = compute_iou(pred_mask, gt_mask)
dice_score = compute_dice(pred_mask, gt_mask)
precision, recall = compute_precision_recall(pred_mask, gt_mask)

# 提取轮廓点
pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(pred_contours) > 0 and len(gt_contours) > 0:
    hausdorff_dist = compute_hausdorff(pred_contours[0].reshape(-1, 2), gt_contours[0].reshape(-1, 2))
else:
    hausdorff_dist = -1  # 无法计算

# 打印评估结果
print(f"IoU Score: {iou_score:.4f}")
print(f"Dice Score: {dice_score:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
print(f"Hausdorff Distance: {hausdorff_dist:.4f}")

# 可视化：叠加预测结果
alpha = 0.5
overlay = cv2.addWeighted(pred_mask.astype(np.uint8) * 255, alpha, gt_mask.astype(np.uint8) * 255, 1 - alpha, 0)
cv2.imshow("Overlay", overlay)

# 可视化：轮廓对比
cv2.drawContours(image, pred_contours, -1, (0, 255, 0), 2)  # 绿色：预测
cv2.drawContours(image, gt_contours, -1, (0, 0, 255), 2)  # 红色：真实
cv2.imshow("Contours", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
