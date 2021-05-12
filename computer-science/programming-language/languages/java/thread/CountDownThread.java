/**
 * Thread - run の実装
 */
class CountDownThread extends Thread {
	private String name;

	public CountDownThread(String name){
		this.name = name;
	}

	public void run(){
		for (int i = 3; i>=0; i--){
			try{
				sleep(1000);
			}catch(InterruptedException e){}
			System.out.println(name + "i");
		}
	}

	public static void main(String[] args){
		CountDownThread t1 = new CountDownThread("Thread 1");
		CountDownThread t2 = new CountDownThread("Thraed 2");

		t1.start();
		t2.start();
	}

}
