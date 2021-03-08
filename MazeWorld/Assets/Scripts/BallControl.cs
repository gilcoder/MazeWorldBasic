using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using ai4u.ext;
using ai4u;

public class BallControl : MonoBehaviour, IAgentResetListener
{
    private TouchRewardFunc touchEvent;
    public DPRLAgent agent;
    // Start is called before the first frame update
    void Start()
    {
        touchEvent = GetComponent<TouchRewardFunc>();
        agent.AddResetListener(this);
    }

    // Update is called once per frame
    void Update()
    {
        if (touchEvent.wasTouched(agent)) {
            gameObject.SetActive(false);
        }
    }

    public virtual void OnReset(Agent agent) {
        gameObject.SetActive(true);
    } 
}
