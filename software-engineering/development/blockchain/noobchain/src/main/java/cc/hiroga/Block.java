package cc.hiroga;

import java.util.Date;

public class Block {
    public String hash;
    public String previousHash;
    private String data; // 今回はシンプルに任意のテキストをデータとして扱う
    private long timeStamp; // エポック秒

	public Block(String data, String previousHash ) {
		this.data = data;
		this.previousHash = previousHash;
		this.timeStamp = new Date().getTime();
		this.hash = calculateHash();
	}

    public String calculateHash() {
        System.out.println("calculateHash() Called!");
        String calculatedhash = StringUtil.applySha256( 
                previousHash +
                Long.toString(timeStamp) +
                data 
                );
        return calculatedhash;
    }
}