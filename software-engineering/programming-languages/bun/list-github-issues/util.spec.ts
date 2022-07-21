// 日本語の入った関数をimportしようとすると、`SyntaxError: Unexpected token '}'. Expected 'as' after the module export name string.`
// import { getTwitterId, getSessionAbstract, getセッションについての補足情報 } from "./util"
import { getJukoushaRole, getJukoushaStartupPhase, getSessionAbstract, getSessionFormat, getSessionHosoku, getSessionTopic, getToudanshaStartupPhase, getTwitterId } from "./util"

const body = "### AWS 行動規範 (Code of Conduct) への同意 (必須)\n\n- [X] 私は、[AWS 行動規範 (Code of Conduct)](https://aws.amazon.com/jp/codeofconduct/) を確認し、同意しました\n\n### Twitter ID (必須)\n\nmats16k\n\n### セッションタイトル (必須)\n\n- [X] セッションのタイトルをイシュー件名に最大40文字程度で入力しました\n\n### セッションのアブストラクト (最大250文字) (必須)\n\nてすとです\n\n### セッションについての補足情報 (最大800文字) (任意)\n\nなんかい感じに話す\n\n### 登壇者の所属するスタートアップのフェーズ （必須）\n\nVC / 投資家\n\n### 想定受講者のスタートアップのフェーズ (複数選択可) (必須)\n\nSeed\n\n### 想定受講者の開発対象やロール・役割 (複数選択可) (必須)\n\nその他のロール\n\n### セッションのトピック (複数選択可) (必須)\n\nその他のトピック\n\n### セッションのフォーマット (必須)\n\nスライドを映しながら話す、通常セッション"

console.log(getTwitterId(body) === "mats16k")
console.log(getSessionAbstract(body) === "てすとです")
console.log(getSessionHosoku(body) === "なんかい感じに話す")
console.log(getToudanshaStartupPhase(body) === "VC / 投資家")
console.log(getJukoushaStartupPhase(body) === "Seed")
console.log(getJukoushaRole(body) === "その他のロール")
console.log(getSessionTopic(body) === "その他のトピック")
console.log(getSessionFormat(body) === "スライドを映しながら話す、通常セッション")
