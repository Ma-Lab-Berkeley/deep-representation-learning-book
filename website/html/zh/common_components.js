/* Chinese language components for common.js */
(function() {
    window.BOOK_COMPONENTS = {
        // Navigation links
        nav: {
            aiTools: "AI 工具",
            aiHelpers: "AI 助手",
            community: "课程、讲座与社区贡献",
            contributors: "编者 / 参编者",
            howToContribute: "如何贡献？",
        },

        // Table of Contents
        toc: {
            preface: "第一版序言",
            prefaceV2: "第二版序言",
            chapter: "第",
            appendix: "附录",
            chapters: {
                1: { title: "第一章", subtitle: "概述" },
                2: { title: "第二章", subtitle: "学习线性和独立结构" },
                3: { title: "第三章", subtitle: "通过去噪追求低维分布" },
                4: { title: "第四章", subtitle: "通过有损压缩的表示学习" },
                5: { title: "第五章", subtitle: "作为展开优化的深度表示" },
                6: { title: "第六章", subtitle: "一致性和自洽性表示" },
                7: { title: "第七章", subtitle: "基于低维分布的推断" },
                8: { title: "第八章", subtitle: "真实世界数据与任务的表示学习" },
                9: { title: "第九章", subtitle: "开放问题与方向" },
            },
            appendices: {
                A: { title: "附录A", subtitle: "优化方法" },
                B: { title: "附录B", subtitle: "熵、扩散、去噪和有损编码" },
            },
        },

        // UI Labels
        ui: {
            bookTitle: "深度表征学习的原理与实践",
            bookSubtitle: "或 记忆的数学理论",
            langLabel: "ZH",
            brandHref: "index.html",
            searchPlaceholder: "搜索页面…",
            menu: "菜单",
            github: "GitHub",
            dateLocale: "zh-CN",
            lastUpdatedTemplate: "最后更新: {date}",
            footer:
                "© {year} Sam Buchanan, Druv Pai, Peng Wang, and Yi Ma. 保留所有权利。",
        },

        // Language options
        languages: {
            en: "English",
            zh: "中文",
        },

        // AI Chat interface
        chat: {
            title: "询问AI",
            clear: "清除",
            close: "关闭",
            send: "发送",
            feedback: "反馈",
            save: "保存",
            chatWithAI: "与AI聊天",
            includeSelection: "包含当前文本选择",
            selectionEmpty: "在页面中选择文本以将其作为上下文包含。",
            placeholder:
                '询问关于此页面的问题…\n\n您也可以通过添加以下内容来询问特定内容：\n@章节（例如"@3"）、@章节.小节（例如"@3.1"）、@章节.小节.子小节（例如"@3.1.2"）\n@附录（例如"@A"）、@附录.小节（例如"@A.1"）、@附录.小节.子小节（例如"@A.1.2"）',
            systemPrompt:
                "您是帮助《深度表征学习的原理与实践》一书读者的AI助手。请清晰简洁地回答。如果相关，请指向当前页面的章节或标题。",
            askAITitle: "询问AI关于此页面",
            modelPicker: {
                title: "选择AI模型",
                options: [
                    {
                        id: "original",
                        text: "🤖 BookQA-7B",
                        description: "原始模型",
                        backgroundColor: "#2196F3",
                        color: "white"
                    },
                    {
                        id: "rag",
                        text: "🧠 BookQA-7B+RAG",
                        description: "RAG增强模型",
                        backgroundColor: "#4CAF50",
                        color: "white"
                    }
                ]
            },
            tooltips: {
                feedback: "提供反馈",
                clear: "清除会话",
                save: "保存聊天记录",
                close: "关闭",
            },
            alerts: {
                noChatHistory: "没有可保存的聊天记录。",
                saveFailed: "保存聊天记录失败。请重试。",
            },
            feedbackNotice: {
                title: "反馈指引",
                bodyMd: `
我们重视您对 BookQA AI 助手的反馈，希望了解您的使用体验！

### 免责声明
您的查询会在我们的本地服务器上匿名记录以便故障排除。为了保护您的隐私，请避免发送敏感信息。

### 🐛 错误报告
如果您在使用 AI 助手时遇到任何问题，请在我们的 GitHub 仓库中报告：

- [报告错误](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/issues)
- 包含您提出的问题和 AI 的回复
- 描述预期与实际行为的差异

### 💡 功能请求
对改进 AI 助手有想法？我们很乐意听取您的建议：

- [提出新的议题进行讨论](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/issues)
- 建议新功能或能力
- 分享尚未得到良好支持的使用案例

### 📚 一般反馈
对于关于本书或本网站的更广泛反馈：

- [贡献指南](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book#making-a-contribution)
- 给作者发邮件（联系信息见书中）

**感谢您帮助我们改进 BookQA AI 助手！**
        `,
            },
        },

        // Sidebar sections
        sidebar: {
            search: "搜索",
            navigation: "导航",
            tableOfContents: "目录",
        },

        // Landing page content
        landing: {
            hero: {
                title: "深度表征学习的原理与实践",
                bookSubtitle: "或 记忆的数学理论",
                authors: "Sam Buchanan · Druv Pai · Peng Wang · Yi Ma",
                subtitle:
                    "一本完全开源的现代教科书，探讨深度神经网络为何以及如何从高维真实世界数据中学习紧凑且信息丰富的表示。",
                buttons: {
                    readHtml: "阅读本书 (HTML)",
                    readPdf: "阅读本书 (PDF)",
                    readPdfZh: "阅读本书 (PDF-ZH)",
                    github: "GitHub 仓库",
                },
                cover: {
                    alt: "书籍封面：深度表征学习的原理与实践",
                    title: "阅读本书",
                    version: "第二版\n发布于 2026年3月1日",
                },
            },
            sections: {
                about: {
                    title: "关于本书",
                    content: `
 在当前深度学习，特别是"生成式人工智能"时代，人们在训练超大型生成模型方面投入了大量资源。迄今为止，这些模型一直是难以理解的"黑盒子"，因为它们的内部机制不透明，导致在可解释性、可靠性和可控性方面存在困难。自然而然地，这种缺乏理解的情况既带来了炒作，也带来了恐惧。

 这本书试图"打开黑盒子"，通过表示学习的视角来理解大型深度网络的机制，表示学习是深度学习模型经验能力的一个主要因素——可以说是最重要的一个因素。本书的简要概述如下。第1章将总结贯穿全书的主线。第2、3、4、5、6章将通过优化和信息论来解释现代神经网络架构的设计原则，将架构开发过程（长期以来被描述为某种"炼金术"）简化为在引入基本原理后的本科水平线性代数和微积分练习。第7章和第8章将讨论这些原理的应用，以更范式化的方式解决问题，获得设计上高效、可解释且可控的新方法和模型，但功能不逊于——有时甚至超过——它们所类似的黑盒模型。第9章将讨论深度学习的潜在未来方向、表示学习的作用以及一些开放问题。

 本书面向具有线性代数、概率论和机器学习背景的高年级本科生或研究生一年级学生。对于数学思维较强的学生，本书应该适合作为深度学习的第一门课程，但拥有一些深度学习的初步表面知识可能有助于更好地理解书中讨论的观点和技术。

 由于本书的时效性，以及深度学习在未来几年可能具有的普遍性，我们决定让本书完全开源，并欢迎学科专家的贡献。源代码可在[GitHub](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book)上获取。在深度表示学习方面，肯定有许多我们在本书中没有涵盖的主题；如果您是专家并认为缺少某些内容，您可以[告诉我们](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book?tab=readme-ov-file#raising-an-issue)或[自己贡献](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book#making-a-contribution)。我们将努力为新贡献保持类似的质量标准，并在[贡献者页面](contributors.html)中认可贡献。`,
                },
                acknowledgements: {
                    title: '致谢',
                    paragraphs: [
                        '本书主要基于过去八年中开发的研究成果。感谢加州大学伯克利分校（2018年）和香港大学（2023年）的慷慨资助，马毅教授得以在过去八年中开始并专注于这个令人兴奋的新研究方向。在这些年中，与此研究方向相关的，马毅教授和他在伯克利的研究团队得到了以下研究资助的支持：',
                        '没有这些研究项目的财政支持，本书就不可能完成。作者们从参与这些项目的同事和学生的研究成果中获得了巨大的启发。'
                    ],
                    grants: [
                        '多大学*THEORINET*深度学习基础项目，由西蒙斯基金会和美国国家科学基金会联合资助（DMS资助 #2031899）',
                        '通过最小化率缩减的*闭环数据转录*项目，由海军研究办公室资助（资助号 N00014-22-1-2102）；',
                        '针对低维结构的*深度学习原理方法*项目，由美国国家科学基金会资助（CISE资助 #2402951）。'
                    ]
                }
            },
            footer: '© {year} Sam Buchanan, Druv Pai, Peng Wang, and Yi Ma. 保留所有权利。'
        },

        // Contributors page content
        contributors: {
            title: "编者 / 参编者",
            intro: "本书的核心作者和参编者。",
            sections: {
                authors: "作者",
                editors: "编辑",
                contributors: "贡献者",
            },

            people: {
                "sam-buchanan": { desc: "主笔作者。" },
                "druv-pai": { desc: "主笔作者。" },
                "peng-wang": { desc: "作者，中文翻译。" },
                "yi-ma": { desc: "资深作者。" },
                "stephen-butterfill": {
                    desc: "PRs：[#14](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/14)",
                },
                "kerui-min": { desc: "中文翻译。" },
                "jan-cavel": {
                    desc: "罗马尼亚语翻译. PRs：[#16](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/16)",
                },
                "kevin-murphy": {
                    desc: "大量反馈。Issues：[#3](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/3)、[#4](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/4)、[#5](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/5)、[#8](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/8)、[#10](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/10)、[#11](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/11)、[#12](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/12)、[#13](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/13)",
                },
                "tianzhe-chu": { desc: "第8.4节实验。AI 工具。" },
                "shenghua-gao": { desc: "第8.8和8.9节。" },
                "bingbing-huang": { desc: "第8.8和8.9节。" },
                "qing-qu": { desc: "第3.3节。" },
                "shengbang-tong": { desc: "第6.3和8.6节。" },
                "jeroen-van-goey": { desc: "PRs：[#29](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/29)" },
                "chengyu-wang": { desc: "第8.8和8.9节。" },
                "ziyang-wu": { desc: "第8.3节。网站开发。" },
                "jingfeng-yang": { desc: "第8.3节。" },
                "daniel-yeh": { desc: "第8.6和8.7节。" },
                "brent-yi": { desc: "第8.10节。" },
                "yaodong-yu": { desc: "为第五章提供初始文本。" },
                "zibo-zhao": { desc: "第8.8节。" },
            },
        },

        // Community page content
        community: {
            title: "课程、讲座与社区贡献",
            intro: "与本书相关的课程、讲座、教程和其他社区贡献。",
            pending: "敬请期待。",
            sections: {
                tutorials: "教程",
                translations: "社区翻译",
                courses: "课程",
                previousVersions: "历史版本",
            },
            content: {
                courses: `
以下是与本书相关或使用了本书材料的高校课程：

 - 香港大学，DATA8014，2025年秋季
   - 讲座 1（追求智能的本质）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec01.pdf)、[视频 1](https://youtu.be/H6vt_OBO0dk)、[视频 2](https://youtu.be/vcpHffMi6NY)、[视频 3](https://youtu.be/7Q-SYdzeR0M)
   - 讲座 2（解析方法与线性模型）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec02.pdf)、[视频 1](https://youtu.be/ybxj5U1cUdM)、[视频 2](https://youtu.be/iJ3GNlWCc4k)、[视频 3](https://youtu.be/Giz1r7cajPY)
   - 讲座 3（学习线性与混合线性模型）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec03.pdf)、[视频 1](https://youtu.be/ctmN-8p7zI8)、[视频 2](https://youtu.be/NSwSh1quLqA)
   - 讲座 4（深度表示学习的优化基础）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec04.pdf)、[视频](https://youtu.be/XDt6SE3kD58)
   - 讲座 5（通过去噪学习低维分布）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec05.pdf)、[视频 1](https://youtu.be/P4Dw0IG2UL8)、[视频 2](https://youtu.be/thjqRBJc8Zw)
   - 讲座 6（通过有损压缩追求低维结构）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec06.pdf)、[视频 1](https://youtu.be/KWuRFK7JKk4)、[视频 2](https://youtu.be/TpnyTR9Y-m0)
   - 讲座 7（通过最大化率缩减构建深度网络）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec07.pdf)、[视频 1](https://youtu.be/lq_vbDC7Hp0)、[视频 2](https://youtu.be/3t3UrWtsnq4)
   - 讲座 8（通过稀疏率缩减的白盒 Transformer 用于表示学习）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec08.pdf)、[视频 1](https://youtu.be/6Ep6HNZSFVs)、[视频 2](https://youtu.be/HdNKSGjVD5E)
   - 讲座 9（通过压缩闭环转录获得自洽表示）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec09.pdf)、[视频 1](https://youtu.be/IJizSfcw6x8)、[视频 2](https://youtu.be/3QFBEAesvFM)
   - 讲座 10（使用学习到的低维分布进行推断）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec10.pdf)、[视频 1](https://youtu.be/MgDclnr0gY4)、[视频 2](https://youtu.be/u5Gn6-S7jKo)
   - 讲座 11（真实世界数据的表示学习）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec11.pdf)、[视频 1](https://youtu.be/1OP4T89dfIk)、[视频 2](https://youtu.be/Vp6X90ojsPA)
   - 讲座 12（总结与智能研究的未来）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec12.pdf)、[视频 1](https://youtu.be/RJsV40-53WU)、[视频 2](https://youtu.be/2-DYPz6jtlY)
 - 澳门大学，CISC7402，2025年秋季

如果您希望将您的课程列在此处，请到 GitHub 提交 issue，并附上课程名称、学校、学期、链接及简短描述。
        `,
                tutorials: `
下面是一些与本书内容高度相关的教程或长篇讲座：
- [ICCV 2025 教程：从高维数据学习低维模型](https://low-dim-models-tutorials.github.io/iccv2025-tutorial/)
  - 讲座 1（演绎式低维解析模型方法）：[幻灯片](https://low-dim-models-tutorials.github.io/iccv2025-tutorial/assets/slides/ICCV2025_Lec1.pdf)
  - 讲座 2（深度模型中低维表征的涌现）：[幻灯片](https://low-dim-models-tutorials.github.io/iccv2025-tutorial/assets/slides/ICCV2025_Lec3.pdf)
  - 讲座 3（从高维数据学习深度低维模型：理论到实践）：[幻灯片](https://low-dim-models-tutorials.github.io/iccv2025-tutorial/assets/slides/ICCV2025_Lec4.pdf)
  - 讲座 4（面向科学应用的受限与可控扩散模型中的低维结构）：[幻灯片](https://low-dim-models-tutorials.github.io/iccv2025-tutorial/assets/slides/ICCV2025_Lec5.pdf)
  - 讲座 5（ReduNet：通过最大化率缩减构建深度网络）：[幻灯片](https://low-dim-models-tutorials.github.io/iccv2025-tutorial/assets/slides/ICCV2025_Lec6.pdf)
  - 讲座 6（通过稀疏率缩减实现白盒 Transformer）：[幻灯片](https://low-dim-models-tutorials.github.io/iccv2025-tutorial/assets/slides/ICCV2025_Lec7.pdf)
- [IAISS 2025 教程：从高维数据学习低维模型](https://2025.iaiss.cc/)
  - 讲座 1（智能的历史与深度学习的数学原理）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec01.pdf)
  - 讲座 2（学习低维线性与独立结构）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec02.pdf)
  - 讲座 3（通过去噪学习低维分布）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec03.pdf)
  - 讲座 4（通过有损压缩学习低维结构）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec04.pdf)
  - 讲座 5（通过展开优化获得深度表示）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec05.pdf)
  - 讲座 6（通过压缩与优化构建白盒深度网络架构）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec06.pdf)
  - 讲座 7（基于低维结构的推断）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec07.pdf)
  - 讲座 8（通过压缩自编码与转录获得一致与自洽表示）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec08.pdf)
  - 讲座 9（机器智能的未来方向）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec09.pdf)
  - 讲座 10（扩散模型中的低维性：可泛化性）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec10.pdf)
  - 讲座 11（扩散模型中的低维性：可控性与合成数据训练）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec11.pdf)
  - 讲座 12（扩散模型中的低维性：求解逆问题与科学 AI）：[幻灯片](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec12.pdf)
- [CVPR 2024 教程：从高维数据学习低维模型](https://low-dim-models-tutorials.github.io/cvpr2024-tutorial/)
  - 讲座幻灯片：[Dropbox](https://www.dropbox.com/scl/fo/7m57krmeordlohel4qxye/AKho1GYbOe0AbBlKNzm28Vk?rlkey=le2yuel4ipq50xhzxmyxxczxi&e=4&st=s8ndn2ix&dl=0)
        `,
                translations: `
以下是（非官方）社区翻译版本：
- [罗马尼亚语译本](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/book_v1_ro.pdf)，由 [Piatra Institute](https://piatra.institute/) 的 Jan Cavel 贡献。
        `,
                previousVersions: `
- [第一版 (PDF)](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/book_v1.pdf) — "Learning Deep Representations of Data Distributions"，发布于2025年8月18日。
`,
            },
        },

        // AI Helpers page content
        aiHelpers: {
            title: "AI 助手",
            intro:
                "本页提供面向本书的简洁 AI 助手。可试用下方 BookQA，更多即将推出。",
            techDetails:
                "BookQA 系列旨在帮助读者理解一本书的内容。它可以回答与材料相关的问题，并清晰解释关键概念与理论。为构建这些模型，我们首先使用 [EntiGraph](https://arxiv.org/pdf/2409.07431) 通过链接从文本中抽样的实体来生成丰富的与书籍相关的数据。随后，我们在这些数据上对 [Qwen2.5-7B/32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) 进行持续的自回归预训练。我们还在训练中加入了指令跟随数据，使模型在学习书中的新知识的同时，不会遗忘基本的对话能力。",
            lightRAGDetails:
                "LightRAG 使用一套全面的解析管线以细粒度处理课程材料，从讲义幻灯片和补充文档中提取准确的文本片段。除了基础的文本抽取之外，系统还对材料中的文本内容和多模态实体执行复杂的实体与关系抽取。该抽取过程使得可以构建一个细粒度、超大规模的知识图谱，在整个课程范围内系统性地连接概念、定义与上下文信息。 当收到用户查询时，LightRAG 利用这一结构化的知识表示，从全局视角检索最相关的知识片段。通过综合实体及其关系结构中所编码的逻辑抽象，系统能够生成不仅准确且符合语境、同时还能完全追溯到特定来源材料的回答。这种可追溯性确保了透明度，并使用户能够验证生成答案的出处，使得 LightRAG 特别适用于需要严格来源验证的教育场景。 集成了 LightRAG 的 BookQA 7B 模型已嵌入本网站顶栏的 “Ask AI” 按钮中。",
            sections: {
                aiAssistants: "AI 助手",
                aiTools: "AI 工具",
                customizedChatbots: "BookQA 系列",
                lightRAG: "LightRAG",
            },
            assistants: [
                {
                    name: "BookQA-7B-Instruct",
                    affil: "",
                    link: "https://huggingface.co/tianzhechu/BookQA-7B-Instruct",
                },
                {
                    name: "BookQA-32B-Instruct",
                    affil: "",
                    link: "https://huggingface.co/tianzhechu/BookQA-32B-Instruct",
                },
            ],
            // badges removed
        },
    };

    // Helper functions to build navigation and TOC arrays
    window.BOOK_COMPONENTS.buildNavLinks = function() {
        return [
            { label: this.nav.contributors, href: "contributors.html" },
            { label: this.nav.aiHelpers, href: "ai_helpers.html" },
            {
                label: this.nav.howToContribute,
                href: "https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book#making-a-contribution",
                external: true,
            },
            { label: this.nav.community, href: "community.html" },
        ];
    };

    window.BOOK_COMPONENTS.buildTOC = function() {
        return [
            { label: this.toc.preface, href: "Chx1.html" },
            { label: this.toc.prefaceV2, href: "Chx2.html" },
            {
                label: this.toc.chapters[1].title,
                subtitle: this.toc.chapters[1].subtitle,
                href: "Ch1.html",
            },
            {
                label: this.toc.chapters[2].title,
                subtitle: this.toc.chapters[2].subtitle,
                href: "Ch2.html",
            },
            {
                label: this.toc.chapters[3].title,
                subtitle: this.toc.chapters[3].subtitle,
                href: "Ch3.html",
            },
            {
                label: this.toc.chapters[4].title,
                subtitle: this.toc.chapters[4].subtitle,
                href: "Ch4.html",
            },
            {
                label: this.toc.chapters[5].title,
                subtitle: this.toc.chapters[5].subtitle,
                href: "Ch5.html",
            },
            {
                label: this.toc.chapters[6].title,
                subtitle: this.toc.chapters[6].subtitle,
                href: "Ch6.html",
            },
            {
                label: this.toc.chapters[7].title,
                subtitle: this.toc.chapters[7].subtitle,
                href: "Ch7.html",
            },
            {
                label: this.toc.chapters[8].title,
                subtitle: this.toc.chapters[8].subtitle,
                href: "Ch8.html",
            },
            {
                label: this.toc.chapters[9].title,
                subtitle: this.toc.chapters[9].subtitle,
                href: "Ch9.html",
            },
            {
                label: this.toc.appendices.A.title,
                subtitle: this.toc.appendices.A.subtitle,
                href: "A1.html",
            },
            {
                label: this.toc.appendices.B.title,
                subtitle: this.toc.appendices.B.subtitle,
                href: "A2.html",
            },
        ];
    };

    window.BOOK_COMPONENTS.coverImagePath = "../assets/book-cover.png";
    window.BOOK_COMPONENTS.bookPdfPath = "../assets/book-main_zh.pdf";
})();
