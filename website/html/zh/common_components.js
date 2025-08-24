/* Chinese language components for common.js */
(function(){
  window.BOOK_COMPONENTS = {
    // Navigation links
    nav: {
      aiTools: 'AI 工具',
      contributors: '编者 / 参编者',
      howToContribute: '如何贡献？'
    },

    // Table of Contents
    toc: {
      preface: '前言',
      chapter: '第',
      appendix: '附录',
      chapters: {
        1: { title: '第一章', subtitle: '引言' },
        2: { title: '第二章', subtitle: '学习线性和独立结构' },
        3: { title: '第三章', subtitle: '通过有损压缩追求低维分布' },
        4: { title: '第四章', subtitle: '通过展开优化实现深度表示' },
        5: { title: '第五章', subtitle: '一致性和自洽性表示' },
        6: { title: '第六章', subtitle: '基于低维分布的推断' },
        7: { title: '第七章', subtitle: '真实世界数据的表示学习' },
        8: { title: '第八章', subtitle: '智能研究的未来' }
      },
      appendices: {
        A: { title: '附录A', subtitle: '优化方法' },
        B: { title: '附录B', subtitle: '熵、扩散、去噪和有损编码' }
      }
    },

    // UI Labels
    ui: {
      bookTitle: '学习数据分布的深度表示',
      langLabel: 'EN',
      brandHref: 'index.html',
      searchPlaceholder: '搜索页面…',
      menu: '菜单',
      github: 'GitHub'
    },

    // Language options
    languages: {
      en: 'English',
      zh: '中文'
    },

    // AI Chat interface
    chat: {
      title: '询问AI',
      clear: '清除',
      close: '关闭',
      send: '发送',
      chatWithAI: '与AI聊天',
      includeSelection: '包含当前文本选择',
      selectionEmpty: '在页面中选择文本以将其作为上下文包含。',
      placeholder: '询问关于此页面的问题…\n\n您也可以通过添加以下内容来询问特定内容：\n@章节（例如"@3"）、@章节.小节（例如"@3.1"）、@章节.小节.子小节（例如"@3.1.2"）\n@附录（例如"@A"）、@附录.小节（例如"@A.1"）、@附录.小节.子小节（例如"@A.1.2"）',
      systemPrompt: '您是帮助《学习数据分布的深度表示》一书读者的AI助手。请清晰简洁地回答。如果相关，请指向当前页面的章节或标题。',
      askAITitle: '询问AI关于此页面'
    },

    // Sidebar sections
    sidebar: {
      search: '搜索',
      navigation: '导航',
      tableOfContents: '目录'
    },

    // Landing page content
    landing: {
      hero: {
        title: '数据分布的深度表达学习',
        authors: 'Sam Buchanan · Druv Pai · Peng Wang · Yi Ma',
        subtitle: '一本完全开源的现代教科书，探讨深度神经网络为何以及如何从高维真实世界数据中学习紧凑且信息丰富的表示。',
        buttons: {
          readHtml: '阅读本书 (HTML)',
          readPdf: '阅读本书 (PDF)',
          readPdfZh: '阅读本书 (PDF-ZH)',
          github: 'GitHub 仓库'
        },
        cover: {
          alt: '书籍封面：学习数据分布的深度表示',
          title: '阅读本书',
          version: '版本 1.0\n发布于 2025年8月18日'
        }
      },
      sections: {
        about: {
          title: '关于本书',
          paragraphs: [
            '在当前深度学习，特别是"生成式人工智能"时代，人们在训练超大型生成模型方面投入了大量资源。迄今为止，这些模型一直是难以理解的"黑盒子"，因为它们的内部机制不透明，导致在可解释性、可靠性和可控性方面存在困难。自然而然地，这种缺乏理解的情况既带来了炒作，也带来了恐惧。',
            '这本书试图"打开黑盒子"，通过表示学习的视角来理解大型深度网络的机制，表示学习是深度学习模型经验能力的一个主要因素——可以说是最重要的一个因素。本书的简要概述如下。第1章将总结贯穿全书的主线。第2、3、4、5章将通过优化和信息论来解释现代神经网络架构的设计原则，将架构开发过程（长期以来被描述为某种"炼金术"）简化为在引入基本原理后的本科水平线性代数和微积分练习。第6章和第7章将讨论这些原理的应用，以更范式化的方式解决问题，获得设计上高效、可解释且可控的新方法和模型，但功能不逊于——有时甚至超过——它们所类似的黑盒模型。第8章将讨论深度学习的潜在未来方向、表示学习的作用以及一些开放问题。',
            '本书面向具有线性代数、概率论和机器学习背景的高年级本科生或研究生一年级学生。对于数学思维较强的学生，本书应该适合作为深度学习的第一门课程，但拥有一些深度学习的初步表面知识可能有助于更好地理解书中讨论的观点和技术。',
            '由于本书的时效性，以及深度学习在未来几年可能具有的普遍性，我们决定让本书完全开源，并欢迎学科专家的贡献。源代码可在[GitHub](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book)上获取。在深度表示学习方面，肯定有许多我们在本书中没有涵盖的主题；如果您是专家并认为缺少某些内容，您可以[告诉我们](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book?tab=readme-ov-file#raising-an-issue)或[自己贡献](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book#making-a-contribution)。我们将努力为新贡献保持类似的质量标准，并在[贡献者页面](contributors.html)中认可贡献。'
          ]
        },
        acknowledgements: {
          title: '致谢',
          paragraphs: [
            '本书主要基于过去八年中所得的研究成果。感谢加州大学伯克利分校（2018）和香港大学（2023）慷慨地提供启动经费，使马毅在过去八年能够投身并专注于这一令人振奋的新研究方向。在这些年里，围绕该研究方向，马毅及其在伯克利的团队获得了以下研究项目的资助：',
          ],
          grants: [
            '由 Simons Foundation 和 National Science Foundation (DMS grant \#2031899）共同资助的多个大学联合开展的THEORINET project for the Foundations of Deep Learning项目；',
            '由 Office of Naval Research (grant N00014-22-1-2102) 资助的Closed-Loop Data Transcription via Minimaxing Rate Reduction项目；',
            '由 National Science Foundation (CISE grant \#2402951) 资助的Principled Approaches to Deep Learning for Low-dimensional Structures项目。'
          ]

          [
            '如果没有这些研究项目的资金支持，本书将无法完成。作者从参与这些项目的同事和学生的研究成果中获得了巨大的启发。'
          ]
        }
      },
      footer: '© {year} Sam Buchanan, Druv Pai, Peng Wang, and Yi Ma. 保留所有权利。'
    },

    // Contributors page content
    contributors: {
      title: '作者 / 参编者',
      intro: '本书的核心作者和参编者',
      sections: {
        coreTeam: '核心编辑团队',
        contributors: '参编者'
      },
      badges: {
        author: '作者',
        leadEditor: '主编',
        seniorAuthor: '资深作者',
        website: '网站',
        chineseTranslation: '中文翻译',
        aiHelper: 'AI助手',
        chapter4: '第四章'
      },
      footer: '© {year} Sam Buchanan, Druv Pai, Peng Wang, and Yi Ma. 保留所有权利。'
    }
  };

  // Helper functions to build navigation and TOC arrays
  window.BOOK_COMPONENTS.buildNavLinks = function() {
    return [
      { label: this.nav.contributors, href: 'contributors.html' },
      { label: this.nav.howToContribute, href: 'https://github.com/Ma-Lab-Berkeley/ldrdd-book#making-a-contribution', external: true }
    ];
  };

  window.BOOK_COMPONENTS.buildTOC = function() {
    return [
      { label: this.toc.preface, href: 'Chx1.html' },
      { label: this.toc.chapters[1].title, subtitle: this.toc.chapters[1].subtitle, href: 'Ch1.html' },
      { label: this.toc.chapters[2].title, subtitle: this.toc.chapters[2].subtitle, href: 'Ch2.html' },
      { label: this.toc.chapters[3].title, subtitle: this.toc.chapters[3].subtitle, href: 'Ch3.html' },
      { label: this.toc.chapters[4].title, subtitle: this.toc.chapters[4].subtitle, href: 'Ch4.html' },
      { label: this.toc.chapters[5].title, subtitle: this.toc.chapters[5].subtitle, href: 'Ch5.html' },
      { label: this.toc.chapters[6].title, subtitle: this.toc.chapters[6].subtitle, href: 'Ch6.html' },
      { label: this.toc.chapters[7].title, subtitle: this.toc.chapters[7].subtitle, href: 'Ch7.html' },
      { label: this.toc.chapters[8].title, subtitle: this.toc.chapters[8].subtitle, href: 'Ch8.html' },
      { label: this.toc.appendices.A.title, subtitle: this.toc.appendices.A.subtitle, href: 'A1.html' },
      { label: this.toc.appendices.B.title, subtitle: this.toc.appendices.B.subtitle, href: 'A2.html' }
    ];
  };

  window.BOOK_COMPONENTS.coverImagePath = "../book-cover.png";
})();
