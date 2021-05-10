public class Nano{
	public static void main(String[] args){
		try{
			long l1 = System.nanoTime();
			Thread.sleep(3000);
			long l2 = System.nanoTime();
			System.out.println(l2-l1);
		}catch(Exception e){
			e.printStackTrace();
		}
	}
}
