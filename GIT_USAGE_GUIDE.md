# Git 和 GitHub 使用指南

## 📚 Git vs GitHub

### Git（本地版本控制系统）
- **是什么**：一个**分布式版本控制系统**，安装在你的电脑/服务器上
- **作用**：在本地管理代码的版本历史、分支、提交等
- **特点**：完全本地化，不需要网络就能工作
- **类比**：就像你本地的"时光机"，可以随时回到代码的任何一个历史版本

### GitHub（远程代码托管平台）
- **是什么**：一个**在线代码托管服务**，基于Git技术
- **作用**：把你的代码上传到云端，方便备份、协作、分享
- **特点**：需要网络连接，提供网页界面
- **类比**：就像"云盘"，但专门用来存储和管理代码

### 关系
```
本地（Git）  ←→  远程（GitHub）
   ↓              ↓
你的服务器      云端仓库
```

**工作流程**：
1. 在本地用 **Git** 管理代码版本
2. 定期推送到 **GitHub** 备份和分享
3. 可以从 **GitHub** 拉取最新代码到本地

---

## 🌿 分支（Branch）概念

### 什么是分支？
分支就像代码的"平行宇宙"，你可以在不影响主代码的情况下，创建新分支来开发新功能。

### 常见分支类型

#### 1. **main/master**（主分支）
- **作用**：存放**稳定、可用的代码**
- **特点**：通常是最重要的分支，代码应该总是可以运行的
- **建议**：不要直接在主分支上开发新功能

#### 2. **feature/xxx**（功能分支）
- **作用**：开发新功能
- **命名**：`feature/添加记忆功能`、`feature/优化训练流程`
- **流程**：从main创建 → 开发 → 测试 → 合并回main → 删除

#### 3. **bugfix/xxx**（修复分支）
- **作用**：修复bug
- **命名**：`bugfix/修复显存溢出`、`bugfix/修复训练错误`
- **流程**：从main创建 → 修复 → 测试 → 合并回main → 删除

#### 4. **dev/develop**（开发分支）
- **作用**：日常开发的主分支
- **特点**：比main更灵活，可以包含未完成的功能
- **流程**：feature分支合并到dev → dev测试稳定后合并到main

---

## 🔄 日常代码管理流程

### 场景1：开发新功能

```bash
# 1. 确保主分支是最新的
git checkout main
git pull origin main

# 2. 创建新功能分支
git checkout -b feature/新功能名称

# 3. 开发代码，修改文件...

# 4. 提交更改
git add .
git commit -m "添加新功能：xxx"

# 5. 推送到GitHub
git push origin feature/新功能名称

# 6. 在GitHub上创建Pull Request（PR），合并到main

# 7. 合并后，删除本地分支
git checkout main
git pull origin main
git branch -d feature/新功能名称
```

### 场景2：修复bug

```bash
# 1. 从main创建bugfix分支
git checkout main
git pull origin main
git checkout -b bugfix/修复描述

# 2. 修复bug，提交
git add .
git commit -m "修复：xxx问题"

# 3. 推送并创建PR
git push origin bugfix/修复描述

# 4. 合并后清理
git checkout main
git pull origin main
git branch -d bugfix/修复描述
```

### 场景3：日常更新代码

```bash
# 1. 查看当前状态
git status

# 2. 添加更改
git add .

# 3. 提交更改
git commit -m "描述你的更改"

# 4. 推送到GitHub
git push origin main
```

### 场景4：同步远程代码

```bash
# 从GitHub拉取最新代码
git pull origin main
```

---

## 📋 常用Git命令速查

### 查看信息
```bash
git status              # 查看当前状态
git log                 # 查看提交历史
git log --oneline       # 简洁的提交历史
git branch              # 查看所有分支
git branch -a           # 查看所有分支（包括远程）
```

### 分支操作
```bash
git branch               # 查看分支
git branch 分支名        # 创建新分支
git checkout 分支名      # 切换到分支
git checkout -b 分支名   # 创建并切换到新分支
git branch -d 分支名     # 删除分支（已合并）
git branch -D 分支名     # 强制删除分支
```

### 提交操作
```bash
git add .                # 添加所有更改
git add 文件名           # 添加特定文件
git commit -m "消息"     # 提交更改
git push origin 分支名    # 推送到远程
git pull origin 分支名    # 从远程拉取
```

### 撤销操作
```bash
git restore 文件名       # 撤销工作区的更改
git restore --staged 文件名  # 取消暂存
git reset --soft HEAD~1  # 撤销最后一次提交（保留更改）
git reset --hard HEAD~1  # 撤销最后一次提交（丢弃更改，危险！）
```

---

## 🎯 推荐的工作流程

### 简单项目（个人开发）
```
main（主分支）
  ↓
直接在主分支上开发、提交、推送
```

### 中等项目（有功能开发）
```
main（稳定版本）
  ↑
feature/功能1  →  开发完成 →  合并
feature/功能2  →  开发完成 →  合并
```

### 复杂项目（团队协作）
```
main（生产环境）
  ↑
dev（开发环境）
  ↑
feature/功能1
feature/功能2
bugfix/修复1
```

---

## 💡 最佳实践

### 1. 提交信息规范
```bash
# 好的提交信息
git commit -m "添加：实现记忆检索功能"
git commit -m "修复：解决训练时的显存溢出问题"
git commit -m "优化：提升模型推理速度"

# 不好的提交信息
git commit -m "更新"
git commit -m "修复bug"
git commit -m "asdf"
```

### 2. 分支命名规范
- `feature/功能名称` - 新功能
- `bugfix/问题描述` - 修复bug
- `hotfix/紧急修复` - 紧急修复
- `refactor/重构内容` - 代码重构

### 3. 定期推送
- 每天结束工作前推送一次
- 完成一个功能后立即推送
- 重要更改后立即推送

### 4. 保持主分支稳定
- 主分支的代码应该总是可以运行的
- 新功能在分支中开发，测试通过后再合并

---

## 🔐 安全建议

### Token管理
- ✅ Token保存在 `~/.git-credentials`（用户级别，安全）
- ✅ 文件权限600，只有你可以访问
- ⚠️ 如果担心安全，可以删除文件，每次手动输入
- 💡 推荐使用SSH key方式（更安全）

### 敏感信息
- ❌ 不要提交密码、API密钥等到Git
- ✅ 使用 `.gitignore` 忽略敏感文件
- ✅ 使用环境变量或配置文件（不提交到Git）

---

## 📖 总结

### 什么时候用Git？
- **总是**：所有代码管理都用Git
- 本地版本控制、分支管理、历史记录

### 什么时候用GitHub？
- **备份**：定期推送代码到GitHub备份
- **协作**：多人协作开发
- **分享**：开源项目或代码分享

### 分支策略
- **个人项目**：可以直接在main上开发
- **有功能开发**：使用feature分支
- **团队项目**：使用dev + feature分支策略

---

## 🚀 快速开始

对于你的项目，推荐的工作流程：

```bash
# 日常开发
git add .
git commit -m "描述更改"
git push origin main

# 开发新功能
git checkout -b feature/功能名
# ... 开发 ...
git add .
git commit -m "添加：功能描述"
git push origin feature/功能名
# 在GitHub创建PR合并到main
```

记住：**Git是工具，GitHub是平台。先用Git管理本地代码，再推送到GitHub备份和分享。**

