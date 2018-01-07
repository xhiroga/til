import java.io.*;

public class TempF{
	public static void main(String[] args){
		try {
			File temp = File.createTempFile("pre","suff");
			String tempPath = temp.getPath();
			System.out.println("一時ファイルパス:" + tempPath);

			Thread.sleep(3000);
		}catch(Exception e){
			e.printStackTrace();
		}
	}

}
