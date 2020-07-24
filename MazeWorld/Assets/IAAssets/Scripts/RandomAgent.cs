using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using ai4u;

public class RandomAgent : MonoBehaviour
{

    private float fx;
    private float fy;

    private Rigidbody mRigidbody;

    private Vector3[] dir;
    private Vector3 TopLeftCorner;
    private Vector3 LowerRightCorner;

    //public GameObject TopLeftCornerObject;
    //public GameObject LowerRightCornerObject;


    public float speed = 10;
    private Vector3 mPos;

    // Start is called before the first frame update
    void Start()
    {
        //TopLeftCorner = TopLeftCornerObject.transform.position;
        //LowerRightCorner = LowerRightCornerObject.transform.position;
        
        dir = new Vector3[]{Vector3.forward, Vector3.forward * -1, Vector3.left, Vector3.right};
        mRigidbody = GetComponent<Rigidbody>();
        mPos = transform.localPosition;
    }

    public void Respawn(){
        gameObject.SetActive(true);
        mRigidbody.velocity = Vector3.zero;
        transform.localPosition = mPos;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
    }
}
