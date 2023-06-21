import styles from "./page.module.css";
import WithApollo from "./components/WithApollo";
import User from "./components/User";

export default function Home() {
  return (
    <main className={styles.main}>
      <WithApollo>
        <User />
      </WithApollo>
    </main>
  );
}
