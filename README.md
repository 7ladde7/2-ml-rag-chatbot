# Agentic RAG Chatbot
Чат-бот, который ответит на вопрос по **PDF**-документу, содержащийся в вашем сообщении в виде ссылки.

Видео-демо:
<p align="center">
  <img src="./demo.gif" alt="Demo" width="500">
</p>

## Архитектура
**Основной стэк:** Python, LangChain, FAISS, RAGAS, OpenChat, Docker

```mermaid
flowchart LR

    A([Вопрос пользователя])
    B[RAG-агент]
    C[Hybrid Retrieval]
    D[Reranker]
    E[Топ-4 результата]
    F{Релевантные?}
    G[Ответ пользователю]
    H[Перефразирование вопроса]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F

    F -->|Да| G
    F -->|Нет| H

    H --> C
```

В чат-боте спользуется LangChain-агент c двумя тулами:
1. Скачивание документа с последующим гибридным поиском (**Hybrid Retrieval**: **BM25** + **FAISS embeddings** + **Reranker**)
2. Проверка релевантности полученного контекста

В качестве **LLM** используется **gpt-5.4-mini**, эмбеддинги получаем через **bge-large-en-v1.5** (лучше всего использовать англоязычные документы), а в роли **Reranker** кросс-энкодер **ms-marco-MiniLM-L-6-v2**.

## Установка
```bash
git clone https://github.com/7ladde7/ml-rag-chatbot.git
cd ml-rag-chatbot
```

Переименуйте файл ```.env.sample``` в  ```.env```, задав в нем переменную ```OPENROUTER_API_KEY```.

Запуск в **Docker**:
```bash
docker compose up --build

# метрики (backend)
python -m evals
```
Открыть чат в браузере: [http://localhost:3000](http://localhost:3000)

## Метрики

Для сравнения с гибридным поиском (**Hybrid Retrieval**) в роли бейзлайна используется агент с **BM25** поиском. Метрики снимаются с помощью **RAGAS**.

Ключевые результаты:
| Тип | Context Recall | Context Precision | Faithfulness |
| :--- | :--- | :--- | :--- |
| BM25 | 0.6250 | 0.7472 | 0.7449 |
| **Hybrid Retrieval** | **0.8583** | **0.9639** | **0.8525** |

## Лицензия
**MIT**

Проект создан исключительно для образовательных целей.