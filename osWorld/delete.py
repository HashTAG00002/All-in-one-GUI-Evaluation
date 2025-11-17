import os
import json
import shutil
from pathlib import Path

def check_traj_jsonl(id_folder_path):
    """检查traj.jsonl文件是否包含client error响应"""
    traj_file = id_folder_path / "traj.jsonl"
    if not traj_file.exists():
        return False
    
    try:
        with open(traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get('response') == 'client error':
                        return True
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"读取文件 {traj_file} 时出错: {e}")
    
    return False

def check_traj_jsonl2(id_folder_path):
    """检查traj.jsonl文件是否包含client error响应"""
    traj_file = id_folder_path / "traj.jsonl"
    if not traj_file.exists():
        return False
    
    try:
        with open(traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("Error") is not None:
                        return True
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"读取文件 {traj_file} 时出错: {e}")
    
    return False

def check_result_txt(id_folder_path):
    """检查是否缺少result.txt文件"""
    result_file = id_folder_path / "result.txt"
    return not result_file.exists()

def find_folders_to_delete(root_dir):
    """找到所有需要删除的ID文件夹"""
    root_path = Path(root_dir)
    folders_to_delete = []
    
    if not root_path.exists():
        print(f"目录 {root_dir} 不存在!")
        return folders_to_delete
    
    # 遍历domain目录
    for domain_dir in root_path.iterdir():
        if not domain_dir.is_dir():
            continue
            
        print(f"正在检查domain: {domain_dir.name}")
        
        # 遍历ID目录
        for id_dir in domain_dir.iterdir():
            if not id_dir.is_dir():
                continue
                
            # 检查条件1：traj.jsonl包含client error
            has_client_error = check_traj_jsonl(id_dir)

            # 检查条件1：traj.jsonl包含error
            has_error = check_traj_jsonl2(id_dir)
            
            # 检查条件2：缺少result.txt
            missing_result = check_result_txt(id_dir)
            
            # 如果满足任一条件（OR关系）
            if has_client_error or missing_result or has_error:
                reason = []
                if has_client_error:
                    reason.append("traj.jsonl包含client error")
                if has_error:
                    reason.append("traj.jsonl包含error")
                if missing_result:
                    reason.append("缺少result.txt文件")
                
                folders_to_delete.append({
                    'path': id_dir,
                    'domain': domain_dir.name,
                    'id': id_dir.name,
                    'reason': '; '.join(reason)
                })
    
    return folders_to_delete

def main():
    # 设置要检查的根目录路径
    #root_directory = input("请输入要检查的根目录路径: ").strip()
    root_directory = "/home/ubuntu/osWorld/results_qwen/pyautogui/screenshot/Qwen2.5-VL-7B-Instruct"
    print("正在扫描目录...")
    folders_to_delete = find_folders_to_delete(root_directory)
    
    if not folders_to_delete:
        print("没有找到需要删除的文件夹!")
        return
    
    # 打印统计信息
    print(f"\n找到 {len(folders_to_delete)} 个需要删除的ID文件夹:")
    print("-" * 80)
    
    for folder_info in folders_to_delete:
        print(f"Domain: {folder_info['domain']}")
        print(f"ID: {folder_info['id']}")
        print(f"路径: {folder_info['path']}")
        print(f"原因: {folder_info['reason']}")
        print("-" * 40)
    
    # 询问用户确认
    confirmation = input(f"\n确认要删除这 {len(folders_to_delete)} 个文件夹吗? (输入 'yes' 确认): ").strip().lower()
    
    if confirmation == 'yes':
        print("\n开始删除文件夹...")
        deleted_count = 0
        
        for folder_info in folders_to_delete:
            try:
                shutil.rmtree(folder_info['path'])
                print(f"✓ 已删除: {folder_info['domain']}/{folder_info['id']}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ 删除失败 {folder_info['domain']}/{folder_info['id']}: {e}")
        
        print(f"\n删除完成! 共删除了 {deleted_count} 个文件夹")
    else:
        print("取消删除操作")

if __name__ == "__main__":
    main()