# Git 和 GitHub 设置完整指南

## 📋 需要准备的信息

### 必需信息：
1. **GitHub 用户名**：你的 GitHub 账号用户名
2. **Personal Access Token (PAT)**：你已经注册的令牌
   - 如果忘记了，可以在 GitHub → Settings → Developer settings → Personal access tokens 查看
   - 注意：如果看不到完整token，需要重新生成一个
3. **仓库名称**：你想在 GitHub 上创建的仓库名称（例如：`qqbot-memory-system`）

### 当前已配置：
- ✅ Git 用户信息：`ymdai` / `ymdai@example.com`
- ✅ SSH Key 已存在（可选使用）

---

## 🚀 操作步骤

### 第一步：在 GitHub 创建仓库

1. 登录 GitHub：https://github.com
2. 点击右上角 **+** → **New repository**
3. 填写信息：
   - **Repository name**: `qqbot-memory-system`（或你喜欢的名字）
   - **Description**: `QQ聊天机器人长期记忆系统`
   - **Visibility**: 选择 **Private**（私有）或 **Public**（公开）
   - ⚠️ **不要勾选** "Add a README file"、"Add .gitignore"、"Choose a license"（本地已有）
4. 点击 **Create repository**

### 第二步：使用 Personal Access Token 连接仓库

创建仓库后，GitHub 会显示设置命令。使用以下命令：

```bash
# 1. 添加远程仓库（替换 YOUR_USERNAME 和 YOUR_REPO_NAME）
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 2. 重命名主分支为 main（如果当前不是 main）
git branch -M main

# 3. 推送代码（会要求输入用户名和密码）
git push -u origin main
```

**重要提示**：
- 用户名：输入你的 **GitHub 用户名**
- 密码：输入你的 **Personal Access Token**（不是 GitHub 登录密码！）

### 第三步：配置 Git 凭据存储（避免每次输入）

为了避免每次推送都输入token，可以配置凭据存储：

```bash
# 方式1：使用 Git Credential Helper（推荐）
git config --global credential.helper store

# 然后执行一次 push，输入用户名和token后会自动保存
git push -u origin main
```

或者使用更安全的方式：

```bash
# 方式2：使用缓存（15分钟内有效）
git config --global credential.helper cache

# 方式3：使用内存存储（进程结束即清除）
git config --global credential.helper 'cache --timeout=3600'
```

---

## 🔐 关于 Personal Access Token

### 如果忘记或需要重新生成：

1. 登录 GitHub
2. 点击右上角头像 → **Settings**
3. 左侧菜单选择 **Developer settings**
4. 选择 **Personal access tokens** → **Tokens (classic)**
5. 点击 **Generate new token (classic)**
6. 填写信息：
   - **Note**: `qqbot项目推送`（描述用途）
   - **Expiration**: 选择过期时间（建议90天或自定义）
   - **Select scopes**: 至少勾选 **repo**（完整仓库权限）
7. 点击 **Generate token**
8. ⚠️ **立即复制token**（只显示一次！）

### Token 权限说明：
- **repo**: 完整仓库访问权限（必需）
- **workflow**: 如果需要 GitHub Actions（可选）

---

## 🔄 两种认证方式对比

### 方式A：HTTPS + Personal Access Token（你当前的方式）
- ✅ 简单直接，你已经有了token
- ✅ 适合大多数情况
- ⚠️ 需要定期更新token（如果设置了过期时间）

### 方式B：SSH Key（你已有SSH key）
- ✅ 更安全，无需定期更新
- ✅ 一次配置，长期使用
- ⚠️ 需要将SSH key添加到GitHub

如果你想使用SSH方式，需要：
1. 将你的SSH公钥添加到GitHub（Settings → SSH and GPG keys）
2. 使用 `git@github.com:用户名/仓库名.git` 格式

---

## 📝 完整操作示例

假设你的信息是：
- GitHub用户名：`ymdai`
- 仓库名：`qqbot-memory-system`
- Token：`ghp_xxxxxxxxxxxxxxxxxxxx`（你的token）

```bash
# 1. 进入项目目录
cd /data0/user/ymdai/LLM_memory/qqbot_new

# 2. 提交所有更改
git add -A
git commit -m "初始提交：QQ聊天机器人长期记忆系统"

# 3. 添加远程仓库
git remote add origin https://github.com/ymdai/qqbot-memory-system.git

# 4. 重命名分支
git branch -M main

# 5. 配置凭据存储（可选，避免每次输入）
git config --global credential.helper store

# 6. 推送代码
git push -u origin main
# 用户名：ymdai
# 密码：ghp_xxxxxxxxxxxxxxxxxxxx（粘贴你的token）
```

---

## ❓ 常见问题

### Q1: 推送时提示 "Authentication failed"
- 检查token是否正确复制（不要有多余空格）
- 确认token有 **repo** 权限
- 确认token未过期

### Q2: 提示 "remote origin already exists"
```bash
# 查看现有远程仓库
git remote -v

# 删除后重新添加
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### Q3: 想使用SSH方式
```bash
# 1. 查看SSH公钥
cat ~/.ssh/id_dsa.pub

# 2. 复制公钥内容，添加到GitHub（Settings → SSH and GPG keys）

# 3. 测试连接
ssh -T git@github.com

# 4. 使用SSH URL添加远程仓库
git remote set-url origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
```

---

## ✅ 检查清单

在开始之前，确认你有：
- [ ] GitHub 账号已登录
- [ ] Personal Access Token 已准备好（或知道如何生成）
- [ ] 知道你的 GitHub 用户名
- [ ] 已在 GitHub 创建了仓库
- [ ] 本地代码已准备好提交

完成这些步骤后，你的代码就会上传到 GitHub 了！

