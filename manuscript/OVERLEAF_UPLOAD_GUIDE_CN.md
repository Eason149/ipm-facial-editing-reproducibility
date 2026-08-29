# Overleaf上传与编译说明

## 最快捷的上传方式

1. 登录 [Overleaf](https://www.overleaf.com/)。
2. 点击 **New Project** → **Upload Project**。
3. 上传本目录生成的 `IPM_Manuscript_Draft_CN_Overleaf.zip`。
4. 项目打开后，点击编辑器左下角的**齿轮 Settings**。
5. 打开 **Compiler** 标签，将编译器改为 **XeLaTeX**。
6. 确认 **Main document** 为 `main.tex`，TeX Live选择最新稳定版本。
7. 关闭设置，点击 **Recompile** 右侧的小箭头并选择 **Recompile from scratch**；如果错误面板中显示 **Clear cached files**，也可先点击它。
8. 再次点击 **Recompile**。

## 项目文件

- `main.tex`：中文全文初稿，也是主编译文件。
- `supplementary_materials.tex`：G复现、连续指标EEG、MVPA/RSA与推断边界补充材料。
- `references.bib`：BibTeX参考文献。
- `figures/Figure_2_A2_construct_validation.png`：A2构念验证与身份遗漏敏感性。
- `figures/Figure_3_behavior_cv_folds.png`：参与者分组五折交叉验证。
- `figures/Figure_4_operation_evidence.png`：操作层行为与预定义窗口EEG结果。
- `figures/Figure_5_full_timecourse_boundary.png`：完整0--1000 ms描述性时程及联合检验边界。
- `figures/Figure_6_participant_robustness.png`：参与者系数与留一参与者稳健性。

## 第一次编辑时优先替换

在 `main.tex` 中搜索“待作者补充”，依次填写：

1. 作者、单位和通讯邮箱；
2. 伦理审批编号与知情同意表述；
3. 量表端点、屏幕尺寸、观看距离和按键映射；
4. 数据与代码共享方式；
5. 利益冲突、CRediT作者贡献和生成式AI使用声明。

## 常见问题

- 出现 `CTeX fontset 'fandol' is unavailable in current mode`：说明项目仍在用pdfLaTeX或LaTeX；在左下角齿轮的 **Compiler** 标签中切换为 **XeLaTeX**，清除缓存后重新编译。代码首行的 `% !TeX program = xelatex` 只是编辑器提示，不能替代Overleaf项目设置。
- 中文乱码或字体错误：确认编译器是 **XeLaTeX**，不要用pdfLaTeX或LaTeX。
- 参考文献第一次为空：连续点击两次 **Recompile**；Overleaf会自动运行BibTeX。
- 图片找不到：不要只上传 `main.tex`，应上传整个ZIP，并保留 `figures` 文件夹。
- 后续改成英文投稿稿：可继续使用 `elsarticle` 框架，删除 `ctex` 包并把中文替换成英文；正式投稿前再从IPM官网核对最新作者指南和模板要求。

## 当前稿件的证据边界

本稿把Skin 350--600 ms和Eye 600--1000 ms表述为锁定窗口内的电位差异，并在解释层面谨慎联系到中晚期/晚期评价加工范围；没有写成离散心理阶段或ERP中介行为，也没有声称早--中--晚完整级联、连续神经追踪或跨身份泛化。后续修改时应保留这些边界。
