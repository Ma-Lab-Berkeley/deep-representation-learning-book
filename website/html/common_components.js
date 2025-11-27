/* English language components for common.js */
(function () {
  window.BOOK_COMPONENTS = {
    // Navigation links
    nav: {
      aiTools: "AI Tools",
      aiHelpers: "AI Helpers",
      community: "Courses, Talks, and Community Contributions",
      contributors: "Contributors",
      howToContribute: "How to Contribute?",
    },

    // Table of Contents
    toc: {
      preface: "Preface",
      chapter: "Chapter",
      appendix: "Appendix",
      chapters: {
        1: { title: "Chapter 1", subtitle: "An Informal Introduction" },
        2: {
          title: "Chapter 2",
          subtitle: "Learning Linear and Independent Structures",
        },
        3: {
          title: "Chapter 3",
          subtitle:
            "Pursuing Low-Dimensional Distributions via Lossy Compression",
        },
        4: {
          title: "Chapter 4",
          subtitle: "Deep Representations from Unrolled Optimization",
        },
        5: {
          title: "Chapter 5",
          subtitle: "Consistent and Self-Consistent Representations",
        },
        6: {
          title: "Chapter 6",
          subtitle: "Inference with Low-Dimensional Distributions",
        },
        7: {
          title: "Chapter 7",
          subtitle: "Learning Representations for Real-World Data",
        },
        8: { title: "Chapter 8", subtitle: "Future Study of Intelligence" },
      },
      appendices: {
        A: { title: "Appendix A", subtitle: "Optimization Methods" },
        B: {
          title: "Appendix B",
          subtitle: "Entropy, Diffusion, Denoising, and Lossy Coding",
        },
      },
    },

    // UI Labels
    ui: {
      bookTitle: "Learning Deep Representations of Data Distributions",
      langLabel: "EN",
      brandHref: "index.html",
      searchPlaceholder: "Search pages…",
      menu: "Menu",
      github: "GitHub",
      dateLocale: "en-US",
      lastUpdatedTemplate: "Last Updated: {date}",
      footer:
        "© {year} Sam Buchanan, Druv Pai, Peng Wang, and Yi Ma. All rights reserved.",
    },

    // AI Chat interface
    chat: {
      title: "Ask AI",
      clear: "Clear",
      close: "Close",
      send: "Send",
      feedback: "Feedback",
      save: "Save",
      chatWithAI: "Ask AI",
      includeSelection: "Include current text selection",
      selectionEmpty: "Select text on the page to include it as context.",
      placeholder:
        'Ask a question about this page…\n\nYou can also ask about specific content by appending:\n@chapter (e.g., "@3"), @chapter.section (e.g., "@3.1"), @chapter.section.subsection (e.g., "@3.1.2")\n@appendix (e.g., "@A"), @appendix.section (e.g., "@A.1"), @appendix.section.subsection (e.g., "@A.1.2")',
      systemPrompt:
        "You are an AI assistant helping readers of the book Learning Deep Representations of Data Distributions. Answer clearly and concisely. If relevant, point to sections or headings from the current page.",
      askAITitle: "Ask AI about this page",
      modelPicker: {
        title: "Select AI model",
        options: [
          {
            id: "original",
            text: "🤖 BookQA-7B",
            description: "Original model",
            backgroundColor: "#2196F3",
            color: "white"
          },
          {
            id: "rag",
            text: "🧠 BookQA-7B+RAG", 
            description: "RAG-enhanced model",
            backgroundColor: "#4CAF50",
            color: "white"
          }
        ]
      },
      tooltips: {
        feedback: "Provide Feedback",
        clear: "Clear conversation",
        save: "Save chat history",
        close: "Close",
      },
      alerts: {
        noChatHistory: "No chat history to save.",
        saveFailed: "Failed to save chat history. Please try again.",
      },
      feedbackNotice: {
        title: "Feedback Guidelines",
        bodyMd: `
We value your feedback on the BookQA AI assistants and would love to hear about your experience!

### Disclaimer
Your queries are anonymously logged on our local server for troubleshooting. In order to protect your privacy, please avoid sending sensitive information.

### 🐛 Bug Reports
If you encounter any issues with the AI helpers, please report them on our GitHub repository:

- [Report a Bug](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/issues)
- Include the question you asked and the AI's response
- Describe the expected vs. actual behavior

### 💡 Feature Requests
Have ideas for improving the AI helpers? We'd love to hear them:

- [Raise a new issue to discuss](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/issues)
- Suggest new features or capabilities
- Share use cases that aren't well supported

### 📚 General Feedback
For broader feedback about the book or this website:

- [Contributing Guide](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book#making-a-contribution)
- Email the authors (contact information in the book)

**Thank you for helping us improve the BookQA AI assistants!**
        `,
      },
    },

    // Language options
    languages: {
      en: "English",
      zh: "中文",
    },

    // Sidebar sections
    sidebar: {
      search: "Search",
      navigation: "Navigation",
      tableOfContents: "Table of Contents",
    },

    // Landing page content
    landing: {
      hero: {
        title: "Learning Deep Representations of Data Distributions",
        authors: "Sam Buchanan · Druv Pai · Peng Wang · Yi Ma",
        subtitle:
          "A modern fully open-source textbook exploring why and how deep neural networks learn compact and information-dense representations of high-dimensional real-world data.",
        buttons: {
          readHtml: "Read the Book (HTML)",
          readPdf: "Read the Book (PDF)",
          readPdfZh: "Read the Book (PDF-ZH)",
          github: "GitHub Repository",
        },
        cover: {
          alt: "Book cover: Learning Deep Representations of Data Distributions",
          title: "Read the Book",
          version: "Version 1.0\nReleased August 18, 2025",
        },
      },
      sections: {
        about: {
          title: "About this Book",
          content: `
In the current era of deep learning and especially "generative artificial intelligence", there is significant investment in training very large generative models. Thus far, such models have been "black boxes" that are difficult to understand in the sense that they have opaque internal mechanisms, leading to difficulties in interpretability, reliability, and control. Naturally, this lack of understanding has led to both hype and fear.

This book is an attempt to "open the black box" and understand the mechanisms of large deep networks, through the perspective of representation learning, which is a major factor — arguably the single most important one — in the empirical power of deep learning models. A brief outline of this book is as follows. Chapter 1 will summarize the threads that underlie the whole text. Chapters 2, 3, 4, and 5 will explain the design principles of modern neural network architectures through optimization and information theory, reducing the process of architecture development (long having been described as a sort of "alchemy") to undergraduate-level linear algebra and calculus exercises once the underlying principles are introduced. Chapters 6 and 7 will discuss applications of these principles to solve problems in more paradigmatic ways, obtaining new methods and models which are efficient, interpretable, and controllable by design, and yet no less — sometimes even more — powerful than the black-box models they resemble. Chapter 8 will discuss potential future directions for deep learning, the role of representation learning, as well as some open problems.

This book is intended for older undergraduate students, or initial graduate students, who have some background in linear algebra, probability, and machine learning. This book should be suitable as a first course in deep learning for mathematically-minded students, but it may help to have some initial surface-level knowledge of deep learning to better appreciate the perspectives and techniques discussed in the book.

Due to the timeliness of the book, and the prevalence that deep learning already has and may continue to have in the future, we have decided to make the book completely open-source. We will update it regularly to improve the presentation of the material and reflect advances in the field, and welcome contributions from subject matter experts about relevant topics that we may have missed. The source code is available on [GitHub](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book). If you feel that something is missing or can be improved, you can [let us know](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book?tab=readme-ov-file#raising-an-issue) or [contribute it yourself](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book#making-a-contribution). We will work to keep a similar standard of quality for new contributions, and recognize contributions in [the contributors page](contributors.html).`,
        },
        acknowledgements: {
          title: "Acknowledgements",
          content: `
This book is primarily based on research results that have been developed within the past eight years. Thanks to generous funding from UC Berkeley (2018) and the University of Hong Kong (2023), Yi Ma was able to embark and focus on this new exciting research direction in the past eight years. Through these years, related to this research direction, Yi Ma and his research team at Berkeley have been supported by the following research grants:

- The multi-university *THEORINET* project for the Foundations of Deep Learning, jointly funded by the Simons Foundation and the National Science Foundation (DMS grant #2031899)
- The *Closed-Loop Data Transcription via Minimaxing Rate Reduction* project funded by the Office of Naval Research (grant N00014-22-1-2102);
- The *Principled Approaches to Deep Learning for Low-dimensional Structures* project funded by the National Science Foundation (CISE grant #2402951).

This book would have not been possible without the financial support for these research projects. The authors have drawn tremendous inspiration from research results by colleagues and students who have been involved in these projects.`,
        },
      },
    },

    // Contributors page content
    contributors: {
      title: "Contributors",
      intro: "Core authors and contributors of the book.",
      sections: {
        authors: "Authors",
        editors: "Editors",
        contentContributors: "Content Contributors",
        infraContributors: "Website/Infrastructure Contributors",
      },

      people: {
        "sam-buchanan": { desc: "Lead author." },
        "druv-pai": { desc: "Lead author." },
        "peng-wang": { desc: "Author, Chinese translation." },
        "yi-ma": { desc: "Senior author." },
        "stephen-butterfill": {
          desc: "PRs: [#14](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/14)",
        },
        "kerui-min": { desc: "Chinese translation." },
        "jan-cavel": {
          desc: "Romanian translation. PRs: [#16](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/16)",
        },
        "kevin-murphy": {
          desc: "Extensive feedback. Issues: [#3](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/3), [#4](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/4), [#5](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/5), [#8](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/10), [#10](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/10), [#11](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/11), [#12](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/12), [#13](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/13)",
        },
        "yaodong-yu": { desc: "Provided initial text for Chapter 4." },
        "tianzhe-chu": { desc: "AI tooling." },
        "ziyang-wu": { desc: "Website development." },
        "jeroen-van-goey": { desc: "PRs: [#29](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book/pull/29)"},
      },
    },

    // Community page content
    community: {
      title: "Courses, Talks, and Community Contributions",
      intro: "Courses, talks, tutorials, and other community contributions related to the book.",
      pending: "Coming soon.",
      sections: {
        tutorials: "Tutorials",
        translations: "Community Translations",
        courses: "Courses",
      },
      content: {
        courses: `
The following are university courses related to or using material from the book.

- The University of Hong Kong, DATA8014, Fall 2025
  - Lecture slides: [1](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec01.pdf), [2](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec02.pdf), [3](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec03.pdf), [4](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec04.pdf), [5](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec05.pdf), [6](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec06.pdf), [7](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec07.pdf), [8](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec08.pdf), [9](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec09.pdf), [10](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec10.pdf), [11](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec11.pdf), [12](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/hku_data8014fa25_slides/lec12.pdf)
  - Lecture videos: [1.1](https://youtu.be/H6vt_OBO0dk), [1.2](https://youtu.be/vcpHffMi6NY), [1.3](https://youtu.be/7Q-SYdzeR0M), [2.1](https://youtu.be/ybxj5U1cUdM), [2.2](https://youtu.be/iJ3GNlWCc4k), [2.3](https://youtu.be/Giz1r7cajPY), [3.1](https://youtu.be/ctmN-8p7zI8), [3.2](https://youtu.be/NSwSh1quLqA), [4](https://youtu.be/XDt6SE3kD58), [5.1](https://youtu.be/P4Dw0IG2UL8), [5.2](https://youtu.be/thjqRBJc8Zw), [6.1](https://youtu.be/KWuRFK7JKk4), [6.2](https://youtu.be/TpnyTR9Y-m0), [7.1](https://youtu.be/lq_vbDC7Hp0), [7.2](https://youtu.be/3t3UrWtsnq4), [8.1](https://youtu.be/6Ep6HNZSFVs), [8.2](https://youtu.be/HdNKSGjVD5E), [9.1](https://youtu.be/IJizSfcw6x8), [9.2](https://youtu.be/3QFBEAesvFM&t=6689s), [10.1](https://youtu.be/MgDclnr0gY4), [10.2](https://youtu.be/u5Gn6-S7jKo), [11.1](https://youtu.be/1OP4T89dfIk), [11.2](https://youtu.be/Vp6X90ojsPA), [12.1](https://youtu.be/RJsV40-53WU), [12.2](https://youtu.be/2-DYPz6jtlY)
- University of Macau, CISC7402, Fall 2025
        `,
        tutorials: `
The following are some tutorials, or long-form lectures, about content which is highly relevant to the book.
- [ICCV 2025 Tutorial on Learning Low-Dimensional Models from High-Dimensional Data](https://low-dim-models-tutorials.github.io/iccv2025-tutorial/)
  - Lecture slides: [1](https://low-dim-models-tutorials.github.io/iccv2025-tutorial/assets/slides/ICCV2025_Lec1.pdf), [2](https://low-dim-models-tutorials.github.io/iccv2025-tutorial/assets/slides/ICCV2025_Lec3.pdf), [3](https://low-dim-models-tutorials.github.io/iccv2025-tutorial/assets/slides/ICCV2025_Lec4.pdf), [4](https://low-dim-models-tutorials.github.io/iccv2025-tutorial/assets/slides/ICCV2025_Lec5.pdf), [5](https://low-dim-models-tutorials.github.io/iccv2025-tutorial/assets/slides/ICCV2025_Lec6.pdf), [6](https://low-dim-models-tutorials.github.io/iccv2025-tutorial/assets/slides/ICCV2025_Lec7.pdf)
- [IAISS 2025 Tutorial on Learning Low-Dimensional Models from High-Dimensional Data](https://2025.iaiss.cc/)
  - Lecture slides: [1](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec01.pdf), [2](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec02.pdf), [3](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec03.pdf), [4](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec04.pdf), [5](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec05.pdf), [6](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec06.pdf), [7](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec07.pdf), [8](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec08.pdf), [9](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec09.pdf), [10](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec10.pdf), [11](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec11.pdf), [12](https://pub-8cdab817e1ea4a47805f543e0f3b71eb.r2.dev/iaiss2025_slides/lec12.pdf)
- [CVPR 2024 Tutorial on Learning Low-Dimensional Models from High-Dimensional Data](https://low-dim-models-tutorials.github.io/cvpr2024-tutorial/)
  - Lecture slides: [Dropbox](https://www.dropbox.com/scl/fo/7m57krmeordlohel4qxye/AKho1GYbOe0AbBlKNzm28Vk?rlkey=le2yuel4ipq50xhzxmyxxczxi&e=4&st=s8ndn2ix&dl=0)

        `,
        translations: `
The following are (unofficial) community translations of the book.
- [A Romanian translation](assets/book-main_ro.pdf), contributed by Jan Cavel at the [Piatra Institute](https://piatra.institute/).
`,
      },
    },

    // AI Helpers page content
    aiHelpers: {
      title: "AI Helpers",
      intro: "We provide simple AI assistants tailored to this book.",
      techDetails: `
The BookQA series of models is designed to help readers understand the book's content. It can answer questions about the material and give clear explanations of the key concepts and theories. To build these models, we first use [EntiGraph](https://arxiv.org/pdf/2409.07431) to generate a rich set of book-related data by linking sampled entities from the text. We then continually pre-train [Qwen2.5-7B/32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on this data using auto-regressive training. We also incorporate instruction-following data during training such that the model can learn new knowledge from the book without forgetting basic chatting skills. The BookQA 7B model currently powers the "Ask AI" button in the top bar of this website.
      `,
      lightRAGDetails: `
LightRAG employs a comprehensive parsing pipeline to process course materials with fine granularity, extracting accurate textual fragments from lecture slides and supplementary documents. Beyond basic text extraction, the system conducts sophisticated entity and relation extraction on both textual content and multimodal entities present within the materials. This extraction process enables the construction of a fine-grained, large-scale knowledge graph that systematically interconnects concepts, definitions, and contextual information across the curriculum.
When a user query is received, LightRAG leverages this structured knowledge representation to retrieve the most relevant knowledge fragments from a global perspective. By synthesizing the logical abstractions encoded in both the entities and their relational structures, the system generates responses that are not only accurate and contextually appropriate but also fully traceable back to specific source materials. This traceability ensures transparency and enables users to verify the provenance of generated answers, making LightRAG particularly well-suited for educational scenarios where source validation is critical.
BookQA 7B model with LightRAG is intergrated into the "Ask AI" button in the top bar of this website.
      `,
      sections: {
        aiAssistants: "AI Assistants",
        aiTools: "AI Tools",
        customizedChatbots: "BookQA Series",
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
  window.BOOK_COMPONENTS.buildNavLinks = function () {
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

  window.BOOK_COMPONENTS.buildTOC = function () {
    return [
      { label: this.toc.preface, href: "Chx1.html" },
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

  window.BOOK_COMPONENTS.coverImagePath = "assets/book-cover.png";
  window.BOOK_COMPONENTS.bookPdfPath = "assets/book-main.pdf";
})();
