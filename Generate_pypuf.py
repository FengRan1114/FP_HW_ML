import numpy as np
import cv2
import os
from tqdm import tqdm
import pypuf.simulation
import pypuf.io

# 设置随机种子以确保结果可复现
np.random.seed(42)


def generate_device_puf(device_id):
    """为每个设备生成独特的PUF模型，仅使用兼容旧版pypuf的类型"""
    seed = device_id + 1000  # 确保种子范围足够大

    n = 64  # 挑战位长度
    k = (device_id % 5) + 3  # 3 - 7之间的XOR级数，每个设备不同

    # 使用XOR Arbiter PUF作为唯一兼容的模型
    # 通过不同的k值和种子确保设备间差异
    return pypuf.simulation.XORArbiterPUF(n, k, seed=seed)


def get_device_noise_profile(device_id):
    """为每个设备定义独特的噪声模式"""
    # 噪声类型：高斯、椒盐、泊松、斑点等
    noise_types = ["gaussian", "salt_pepper", "poisson", "speckle"]
    noise_type = noise_types[device_id % len(noise_types)]

    # 噪声强度
    intensity = 0.2 * (device_id % 5 + 1)  # 增大噪声强度范围，0.2 - 1.0之间
    # 纹理频率（用于结构化噪声）
    frequency = 0.1 * (device_id % 10 + 1)

    return {
        "type": noise_type,
        "intensity": intensity,
        "frequency": frequency,
        "seed": device_id + 2000  # 独立的噪声种子
    }


def apply_noise(image, noise_profile):
    """根据噪声配置应用不同类型的噪声"""
    np.random.seed(noise_profile["seed"])
    row, col = image.shape

    if noise_profile["type"] == "gaussian":
        mean = 0
        var = noise_profile["intensity"] * 255
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    elif noise_profile["type"] == "salt_pepper":
        s_vs_p = 0.5
        amount = noise_profile["intensity"]

        # 使用元组索引避免FutureWarning
        out = np.copy(image)
        # 盐噪声
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 255

        # 椒噪声
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0

        return out

    elif noise_profile["type"] == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    elif noise_profile["type"] == "speckle":
        gauss = np.random.randn(row, col) * noise_profile["intensity"]
        noisy = image + image * gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    return image


def challenge_to_image(challenge, width=220, height=200, device_id=None):
    """将二进制挑战转换为具有独特模式的图像"""
    # 使用设备ID作为随机种子，确保每个设备的挑战图像模式不同
    seed = device_id if device_id is not None else 0
    np.random.seed(seed)

    # 创建基于挑战的独特模式，强化噪点效果
    noise_base = np.random.normal(0, 100, (height, width)).astype(np.uint8)
    challenge_matrix = challenge.reshape(-1, int(np.sqrt(len(challenge))))
    challenge_matrix = cv2.resize(challenge_matrix.astype(np.float32), (width, height),
                                  interpolation=cv2.INTER_LINEAR)
    challenge_matrix = (challenge_matrix * 127.5 + 127.5).astype(np.uint8)
    challenge_img = cv2.add(challenge_matrix, noise_base)

    # 应用设备特定的噪声
    noise_profile = get_device_noise_profile(device_id)
    challenge_img = apply_noise(challenge_img, noise_profile)

    return challenge_img


def response_to_image(response, width=220, height=200, device_id=None):
    """将PUF响应转换为具有设备特定特征的图像"""
    # 基础响应值决定整体亮度
    base_value = 127.5 * (response + 1)

    # 创建设备特定的纹理模式，强化噪点效果
    texture = np.random.normal(0, 100, (height, width)).astype(np.uint8)
    response_img = np.ones((height, width)) * base_value + texture

    # 添加设备特定的低频模式，增强区分度
    frequency = 0.05 * (device_id + 1)
    x = np.linspace(0, 2 * np.pi, width)
    y = np.linspace(0, 2 * np.pi, height)
    xx, yy = np.meshgrid(x, y)

    if device_id % 3 == 0:
        device_pattern = np.sin(xx * frequency) * np.cos(yy * frequency) * 40
    elif device_id % 3 == 1:
        device_pattern = np.sin(xx * frequency + yy * frequency) * 40
    else:
        device_pattern = np.cos(xx * frequency) * np.sin(yy * frequency) * 40
    response_img += device_pattern

    # 应用设备特定的噪声
    noise_profile = get_device_noise_profile(device_id)
    response_img = apply_noise(response_img.astype(np.uint8), noise_profile)

    # 归一化到0 - 255范围
    return np.clip(response_img, 0, 255).astype(np.uint8)


def generate_crp_images(device_id, num_images=192, output_dir="puf_images"):
    """为单个设备生成CRP图像，使用DeviceX命名约定"""
    device_name = f"Device{device_id + 1}"  # 命名为Device1 - Device10
    device_dir = os.path.join(output_dir, device_name)
    os.makedirs(device_dir, exist_ok=True)

    # 生成设备PUF模型
    puf = generate_device_puf(device_id)

    # 生成随机挑战
    n = puf.challenge_length
    challenges = pypuf.io.random_inputs(n, num_images, seed=device_id)

    for i in tqdm(range(num_images), desc=f"生成{device_name}的图像"):
        challenge = challenges[i]

        # 获取PUF响应 (-1 或 1)
        response = puf.eval(np.array([challenge]))[0]

        # 创建挑战图像和响应图像
        challenge_img = challenge_to_image(challenge, device_id=device_id)
        response_img = response_to_image(response, device_id=device_id)

        # 保存图像
        cv2.imwrite(os.path.join(device_dir, f"challenge_{i}.png"), challenge_img)
        cv2.imwrite(os.path.join(device_dir, f"response_{i}.png"), response_img)


def main():
    """主函数：生成10个设备的CRP图像"""
    num_devices = 10
    num_images_per_device = 192

    for device_id in range(num_devices):
        generate_crp_images(device_id, num_images_per_device)

    print("所有设备的CRP图像生成完成！")
    print(f"图像保存在: {os.path.abspath('puf_images')}")


if __name__ == "__main__":
    main()
