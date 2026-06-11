# Security Policy

## Scope

This project generates TRON private keys and addresses. A private key controls all assets held by the corresponding address. If a private key is exposed, copied, uploaded, logged, intercepted, or recovered by malware, the funds in that address can be permanently lost.

The project is designed to be auditable and offline-friendly, but no software can make an untrusted computer safe. Review the source code, understand the output format, and run it only in an environment you trust.

## Supported Versions

Security fixes are provided for the latest commit on the default branch and for the latest published release, if a release is available.

Older commits, third-party forks, repackaged binaries, and unofficial downloads are not supported by this project.

## Recommended Safe Usage

- Prefer running from source after reviewing the code.
- Run the generator on a trusted machine, ideally offline.
- Do not run it on shared, remote-controlled, infected, or untrusted computers.
- Do not paste generated private keys into websites, chat apps, cloud notes, screenshots, or online converters.
- Do not upload match output files to cloud storage, GitHub, messaging apps, email, or issue trackers.
- Test a newly generated address with a small amount before transferring meaningful funds.
- After importing the private key into your wallet, securely delete plaintext output files if you no longer need them.
- Treat terminal scrollback, shell history, crash dumps, antivirus quarantine copies, indexing services, and backup tools as possible private-key exposure paths.

## Source Builds vs. Binary Releases

For security-sensitive use, source builds are recommended.

If a binary release is provided, verify that:

- It comes from the official project release page.
- The release notes identify the exact source commit used to build it.
- A SHA256 checksum is provided and matches the downloaded file.
- You trust the build environment and packaging process.

Do not trust binaries mirrored by third parties. This project cannot guarantee the safety of repackaged archives or modified executables.

## Network Behavior

The source code in this repository is intended to generate addresses locally. It should not need network access to search for vanity addresses.

Before using this software with real funds, you are encouraged to audit the code for network APIs, file writes, process spawning, dynamic imports, and bundled binary behavior.

## Private Key Output

Generated matches are written to local output files and include plaintext private keys. Anyone who can read these files can control the corresponding addresses.

Protect these files with the same care as wallet seed phrases:

- Keep them out of synced folders.
- Do not share logs or screenshots that contain them.
- Restrict filesystem permissions where possible.
- Delete or move them to encrypted storage after use.

## Reporting a Vulnerability

Please report security issues privately before public disclosure.

Recommended report contents:

- A clear description of the issue.
- Steps to reproduce.
- The affected commit or release version.
- Operating system, Python version, GPU model, and driver version if relevant.
- Whether the issue can expose, corrupt, or incorrectly generate private keys.

If you cannot find a private contact channel, open a GitHub issue with minimal non-sensitive details and ask for a private follow-up path. Do not include private keys, seed phrases, real wallet addresses with funds, exploit code, or sensitive logs in a public issue.

## Disclosure Expectations

The maintainer will make a best effort to:

- Acknowledge valid reports.
- Investigate the impact.
- Publish a fix or mitigation when practical.
- Credit the reporter if requested and appropriate.

No response-time guarantee is provided.

## Disclaimer

This software is provided without warranty. You are responsible for reviewing, building, running, and securing it. Cryptocurrency transactions are irreversible, and private-key exposure cannot be undone.

---

# 安全策略

## 适用范围

本项目会生成 TRON 私钥和地址。私钥控制对应地址中的全部资产。一旦私钥被泄露、复制、上传、记录、截屏、被恶意软件读取，地址中的资产可能永久丢失。

本项目尽量做到源码可审计、适合离线运行，但任何软件都无法把一台不可信的电脑变成安全环境。请先审计源码，理解输出格式，并只在你信任的环境中运行。

## 支持版本

安全修复优先面向默认分支的最新提交，以及最新发布版本。

旧提交、第三方 fork、重新打包的二进制文件、非官方下载渠道，不属于本项目支持范围。

## 推荐安全用法

- 优先从源码运行，并在运行前审计代码。
- 在可信机器上运行，最好断网运行。
- 不要在共享电脑、远控环境、疑似中毒电脑或不可信电脑上运行。
- 不要把生成的私钥粘贴到网站、聊天软件、云笔记、截图或在线转换工具中。
- 不要把命中结果文件上传到网盘、GitHub、聊天软件、邮件或 issue。
- 新地址正式使用前，先用小额资产测试。
- 私钥导入钱包后，如不再需要明文输出文件，请安全删除。
- 终端滚屏、shell 历史、崩溃转储、杀毒软件隔离区、系统索引和备份软件，都可能成为私钥泄露路径。

## 源码运行与二进制发布

涉及真实资产时，推荐从源码运行。

如果使用二进制发布包，请确认：

- 下载来源是官方项目 release 页面。
- release 说明标明了构建所用的源码 commit。
- 提供了 SHA256 校验值，且与你下载的文件一致。
- 你信任该构建环境和打包流程。

不要信任第三方转载、重新打包或来源不明的可执行文件。本项目无法保证这些文件的安全性。

## 网络行为

本仓库源码的设计目标是在本地生成地址。搜索靓号地址不应需要联网。

在使用本软件管理真实资产前，建议你审计源码中的网络 API、文件写入、进程调用、动态导入和捆绑二进制行为。

## 私钥输出

命中结果会写入本地输出文件，并包含明文私钥。任何能读取这些文件的人，都能控制对应地址。

请像保护助记词一样保护这些文件：

- 不要放在自动同步目录中。
- 不要分享包含私钥的日志或截图。
- 尽可能限制文件权限。
- 使用后删除，或转移到加密存储中。

## 报告漏洞

请优先私下报告安全问题，避免直接公开披露。

建议报告内容包括：

- 问题描述。
- 复现步骤。
- 受影响的 commit 或 release 版本。
- 如相关，请提供操作系统、Python 版本、GPU 型号和驱动版本。
- 说明问题是否可能泄露私钥、破坏私钥、或生成错误地址。

如果找不到私密联系方式，请在 GitHub issue 中提交最少量的非敏感信息，并请求维护者提供私下沟通方式。不要在公开 issue 中包含私钥、助记词、真实有资产的钱包地址、利用代码或敏感日志。

## 披露预期

维护者会尽力：

- 确认有效报告。
- 分析影响范围。
- 在可行时发布修复或缓解方案。
- 在合适且报告者愿意的情况下致谢。

本项目不承诺固定响应时间。

## 免责声明

本软件不提供任何担保。你需要自行负责审计、构建、运行和安全保管。加密货币交易不可逆，私钥泄露无法撤销。
