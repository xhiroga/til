export default function Center({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '30px',
        alignItems: `center`,
        justifyContent: `center`,
        height: `1000px`,
      }}
    >
      {children}
    </div>
  );
}
