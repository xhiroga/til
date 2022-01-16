import java.io.*;
import java.net.*;

public class TestServer{
    public static void main(String[] args) throws Exception {
        try{
            ServerSocket server = new ServerSocket(8001);
            FileOutputStream fos = new FileOutputStream("server_recv.txt");
            System.out.println("クライアント接続待ち");

            Socket socket = server.accept();
            System.out.println("クライアント接続");

            int ch;
            InputStream input = socket.getInputStream();
            while((ch = input.read())!=-1){
                fos.write(ch);
            }
            socket.close();
        } catch (Exception e){
            e.printStackTrace();
        }
    }
}
