import java.io.*;

public class FileRead{

	public static void main(String args[]){

		try{
			FileReader reader = new FileReader("read.txt");

			char buf[] = new char[32];

			while(reader.read(buf)!=-1){
				System.out.println(buf);
			}

			reader.close();
		}catch(IOException e){
		// do nothing
		}
	}
}
