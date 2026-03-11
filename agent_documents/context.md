# エージェント指示書

## コンテキスト（Context）

あなたはプロジェクトのルートディレクトリで動作しています。

## タスク（Your Task）

1. `agent_documents/prd.json` を読み込む。
2. `agent_documents/progress.md` を読み込む。（最初に「コードベースのパターン」を確認すること）
3. 正しいブランチにいるか確認する（存在しない場合は作成し、存在する場合はチェックアウトする）。
4. `status: 未実行` となっているストーリーの中から、最も上位のタスクを選択する。
5. コードベースの調査や実装計画を立てる。必要であれば、Gemini Searchを使い、web検索を行う。
6. **現在のディレクトリ (`.`)** で、そのストーリーを1つだけ実装する。`agent_documents/` 内には実装しないこと。
7. 型チェックとテストを実行する（その言語で利用可能な場合）。
8. コミットを行う： `feat: [ID] - [Title]`
9. `agent_documents/prd.json` を更新する：完了したストーリーの `未実行` を `実行` に設定する。
10. `agent_documents/progress.md` にコードベースのパターン・進捗を追記する。

## 進捗フォーマット（Progress Format）

`agent_documents/progress.md` の末尾に**追記**してください：

例：
```
### [日付] - [ストーリーID]

* 実装内容
* 変更されたファイル
* **知見（Learnings）:**
* 発見されたパターン
* 直面した注意点（ハマりどころ）
```

## コードベースのパターン（Codebase Patterns）

再利用可能なパターンを `agent_documents/progress.md` の**冒頭**に追加してください

例：
```
## Codebase Patterns

* Migrations: IF NOT EXISTS を使用すること
* React: useRef<Timeout | null>(null) を使用すること
```

## Gemini Search（Gemini 検索）

`gemini` は Google Gemini の CLI ツールです。これを使用してウェブ検索を行うことができます。
Task Tool 経由で、`gemini -p 'WebSearch: 自然言語での詳細なクエリ'` を実行してウェブ検索を利用してください。

```bash
gemini -p "WebSearch: <検索クエリ>"

```
