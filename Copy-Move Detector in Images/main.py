import numpy as np
import cv2
import CopyMoveDetect
import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=False, default="result.png")
parser.add_argument("-b", "--blockSize", required=False, default=8, type=int)
parser.add_argument("-r", "--reduce", required=False, default=16, type=int)
parser.add_argument("-q", "--quantization", required=False, default=16, type=int)
parser.add_argument("-cc", "--compareCount", required=False, default=1, type=int)
parser.add_argument("-d", "--differenceThreshold", required=False, default=2, type=int)
parser.add_argument("-n", "--shiftNormThreshold", required=False, default=50, type=int)
parser.add_argument("-c", "--shiftCountThreshold", required=False, default=150, type=int)
args = parser.parse_args()

intro = """
             :oyhddhddhys+:
           sdmmhhhhyhhhdmmdy+-
         smhshhdhysoosyhhhhsdy+	 	$$\   $$\ $$$$$$$$\ $$\   $$\ 
        hmdydho:.`    `.:ydhhmh+ 	$$ | $$  |\__$$  __|$$ |  $$ |
       ymdyds/`          `/dhhmh+	$$ |$$  /    $$ |   $$ |  $$ |
       mhsms/`            `+myhmo	$$$$$  /     $$ |   $$ |  $$ |
       myyms- `hddy  hddy` :mhsms	$$  $$<      $$ |   $$ |  $$ |
       mmymy/ .mydy  mydy` +msmms	$$ |\$$\     $$ |   $$ |  $$ |
       ymymmy/:msdh  dsdy./dhhmh+	$$ | \$$\    $$ |   \$$$$$$  |
        hhohmdymsmh  msmhymysddo 	\__|  \__|   \__|    \______/ 
         smmhhdmsmd  msmmhhmmy+			
           sdmdhohd  doyhmdy+-         Computer Engineering (Forever <3)
              oyyhy  dhys+					

 .-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-.
 |                                                                       |
 |                 COPY-MOVE FORGERY DETECTION TOOL                      |
 |            @AliEmreKeskin - @altunmustafa - @abdllhcay                |
 |        github.com/AliEmreKeskin/Copy-Move-Forgery-Detection           |
 |                                                                       |
 .-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-.

"""

os.system("cls")
print(intro)

#loading source image
img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
height, width = img.shape[:2]
#cv2.imshow("Input file",img)
#cv2.waitKey(1)

print("[+] Input file: {}".format(args.input))
print("[+] Dimensions: {}x{}".format(width, height))

#result=np.copy(img)
result = np.zeros((height,width),np.uint8)

start_time = time.time()
CopyMoveDetect.BlockMatchDCT(img,result,args.blockSize,args.reduce,args.quantization,args.compareCount,args.differenceThreshold,args.shiftNormThreshold,args.shiftCountThreshold)
end_time = time.time()
elapsed_time = end_time - start_time

#cv2.imshow("result",result)
#cv2.waitKey(1)

cv2.imwrite(args.output,result)

print("[+] Done!")
print("[+] Elapsed in: %.2f\n" %elapsed_time)

#cv2.waitKey(0)
#cv2.destroyAllWindows()