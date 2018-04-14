package cc.hiroga;

import java.security.MessageDigest;

public class StringUtil {
    // 文字列をSHA256でダイジェストして返却
    public static String applySha256(String input){
        System.out.println("applySha256("+ input +") Called!");
        try{
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(input.getBytes("UTF-8"));
            StringBuffer hexString = new StringBuffer();
            for (int i = 0; i < hash.length; i++) {
				String hex = Integer.toHexString(0xff & hash[i]);
				if(hex.length() == 1) hexString.append('0');
				hexString.append(hex);
			}
            return hexString.toString();
        }catch(Exception e){
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }
}