import java.io.*;

public class BufferdRead{
	public static void main(String[] args){
		try {
		BufferedReader br = new BufferedReader(new FileReader("read.txt"));
		String str;
		while((str=br.readLine())!=null){
			System.out.println(str);
		}
		}catch(IOException e){
		}
	}
}
