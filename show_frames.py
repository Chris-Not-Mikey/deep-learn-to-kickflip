import cv2



if __name__ == "__main__":


    cap = cv2.VideoCapture('./videos/train/training_video.mp4')
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Read until video is completed
    counter = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        imS = cv2.resize(frame, (960, 540))  
        # Display each frame
        cv2.imshow("video", imS)
        # show one frame at a time
        print("frame: " + str(counter))
        key = cv2.waitKey(0)
        while key not in [ord('q'), ord('k')]:
            key = cv2.waitKey(0)
        # Quit when 'q' is pressed
        if key == ord('q'):
            break

        counter = counter + 1

    # When everything done, release the video capture object
    cap.release()



    print("#########################################")
    print("#########Computation Complete############")
    print("#########################################")