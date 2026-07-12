# 修复 Git Remote 代理地址失败问题

## 问题现象

执行：

```bash
git pull
```

出现类似错误：

```text
fatal: unable to access 'https://mirror.ghproxy.com/https://github.com/MrYuxiangZhu/animal_det.git/': Failed to connect to mirror.ghproxy.com port 443
```

原因是当前仓库的 Git remote 被设置成了不可访问的代理地址：

```text
https://mirror.ghproxy.com/https://github.com/MrYuxiangZhu/animal_det.git
```

当 `mirror.ghproxy.com` 无法访问时，`git pull` 就会失败。

## 自动修复脚本

工程中已经提供自动修复脚本：

```text
script/fix_git_remote.sh
```

## 推荐用法

### 1. 切回 GitHub 官方 HTTPS 地址

推荐先执行：

```bash
bash script/fix_git_remote.sh
```

默认会把 `origin` 改成：

```text
https://github.com/MrYuxiangZhu/animal_det.git
```

脚本会自动测试远端连接：

```bash
git ls-remote origin HEAD
```

如果成功，再执行：

```bash
git pull
```

## SSH 用法

如果你已经配置过 GitHub SSH key，可以执行：

```bash
bash script/fix_git_remote.sh origin ssh
```

它会把 `origin` 改成：

```text
git@github.com:MrYuxiangZhu/animal_det.git
```

然后执行：

```bash
git pull
```

## 指定任意 remote 地址

脚本也支持直接传入完整 remote 地址：

```bash
bash script/fix_git_remote.sh origin https://github.com/MrYuxiangZhu/animal_det.git
```

或者：

```bash
bash script/fix_git_remote.sh origin git@github.com:MrYuxiangZhu/animal_det.git
```

## 参数说明

脚本格式：

```bash
bash script/fix_git_remote.sh [remote_name] [mode_or_url] [repo_path]
```

参数：

```text
remote_name   默认 origin
mode_or_url   默认 github，可选 github / https / ssh / 完整 URL
repo_path     默认 MrYuxiangZhu/animal_det.git
```

示例：

```bash
bash script/fix_git_remote.sh origin github
bash script/fix_git_remote.sh origin ssh
bash script/fix_git_remote.sh origin https MrYuxiangZhu/animal_det.git
```

## 脚本会做什么

脚本会自动：

1. 定位项目根目录；
2. 检查当前目录是否是 Git 仓库；
3. 读取当前 remote 地址；
4. 如果发现 `mirror.ghproxy.com` 或 `ghproxy`，自动替换；
5. 打印当前 remote 配置；
6. 执行远端连接测试；
7. 如果成功，提示可以执行 `git pull`；
8. 如果失败，输出排查建议。

## 推荐处理流程

优先使用 HTTPS：

```bash
bash script/fix_git_remote.sh
git pull
```

如果 HTTPS 失败，尝试 SSH：

```bash
bash script/fix_git_remote.sh origin ssh
git pull
```

如果 SSH 也失败，检查 SSH key：

```bash
ssh -T git@github.com
```

## 如果仍然失败

可能原因：

```text
1. 当前网络无法访问 github.com；
2. DNS 异常；
3. 系统代理配置错误；
4. GitHub 仓库权限问题；
5. SSH key 没有添加到 GitHub；
6. 公司/校园网络限制 GitHub 访问。
```

可以尝试清理 Git 代理：

```bash
git config --global --unset http.proxy || true
git config --global --unset https.proxy || true
```

然后重新测试：

```bash
git ls-remote origin HEAD
```

## 查看当前 remote

```bash
git remote -v
```

正常 HTTPS 应该类似：

```text
origin  https://github.com/MrYuxiangZhu/animal_det.git (fetch)
origin  https://github.com/MrYuxiangZhu/animal_det.git (push)
```

正常 SSH 应该类似：

```text
origin  git@github.com:MrYuxiangZhu/animal_det.git (fetch)
origin  git@github.com:MrYuxiangZhu/animal_det.git (push)
```
