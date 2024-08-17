from skimage.metrics import structural_similarity as ssim
from skimage import io


def compute_ssim(image_path1, image_path2):
    # 读取图像
    img1 = io.imread(image_path1)
    img2 = io.imread(image_path2)

    # 计算 SSIM
    ssim_index, _ = ssim(img1, img2, multichannel=True, full=True, channel_axis=2)

    return ssim_index


if __name__ == "__main__":
    image1_path = r"C:\Users\11504\Desktop\RN-Puzzle\test_results\1_0_gt.png"  # 图像1的路径
    image2_path = r"C:\Users\11504\Desktop\RN-Puzzle\test_results\1_0_output.png"  # 图像2的路径

    ssim_value = compute_ssim(image1_path, image2_path)
    print(f"The SSIM value between the images is: {ssim_value}")
