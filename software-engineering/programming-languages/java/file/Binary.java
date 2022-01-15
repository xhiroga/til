import java.io.*;

public class Binary{
	public static void main(String[] args){
		try{
			FileInputStream input = new FileInputStream("read.txt");
			FileOutputStream output = new FileOutputStream("write.txt");

			byte buf[]=new byte[256];
			int len;
			while((len=input.read(buf))!=-1){
				output.write(buf,0,len);
			}
			output.flush();
			output.close();
			input.close();
		}catch(IOException e){
		}
	}
}
