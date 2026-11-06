"""Native-step v3 reward ledger. No simulator imports: boundary checks run on CPU."""
import copy, math
import numpy as np
VERSION='rpent-skills-v3-20261104'
class RewardLedger:
    def __init__(self,cfg,skill,initial):
        self.cfg=copy.deepcopy(cfg);self.skill=skill;self.initial=copy.deepcopy(initial);self.prev=copy.deepcopy(initial)
        self.step=0;self.peak=initial['lift_m'];self.path=0.;self.descent=0.;self.earned={};self.events=set();self.hold=0;self.stable=0;self.hold_heights=[];self.done=False;self.total=0.;self.release_eligible=False
    def advance(self,m):
        assert not self.done
        c=self.cfg;s=c[self.skill];t=c['success'];self.step+=1;parts={'step':-c['reward']['step_penalty']}
        def credit(name,score,budget):
            score=float(np.clip(score,0,1));old=self.earned.get(name,0.)
            parts[name]=budget*max(0.,score-old);self.earned[name]=max(old,score)
        def event(name,ok,budget):
            if ok and name not in self.events:self.events.add(name);parts[name]=budget
        travel=float(np.linalg.norm(np.array(m['object_pos'][:2])-self.prev['object_pos'][:2]));self.path+=travel
        xy=float(np.linalg.norm(np.array(m['object_pos'][:2])-self.initial['object_pos'][:2]));self.peak=max(self.peak,m['lift_m'])
        parts['horizontal']=-s['horizontal_penalty_per_m']*travel if s.get('horizontal_penalty_enabled') else 0.
        local=m['target_xy_distance']<=s.get('target_radius_m',float('inf'))
        upright=m['tilt_degrees']<=t['upright_degrees']
        quiet=m['linear_speed']<=t['linear_speed_m_s'] and m['angular_speed']<=t['angular_speed_rad_s']
        contact=local and upright and quiet and m['target_supported']
        if m['grasped']:self.release_eligible=bool(contact)
        elif not contact:self.release_eligible=False
        released=not m['grasped'] and m['gripper_opening']>.06
        retreat=m['eef_object_center_distance']>=t['eef_clearance_m']
        stable=contact and released and retreat
        self.stable=self.stable+1 if stable else 0
        phase='approach'
        if self.skill=='pick':
            h=m['lift_m'];lo=s['lift_reward_start_m'];hi=s['lift_reward_max_m']
            if self.earned.get('height',0)>0 and s['descent_penalty_enabled']:
                old=max(0.,self.descent-s['descent_deadband_m']);self.descent+=max(0.,self.prev['lift_m']-h)
                parts['descent']=-s['descent_penalty_per_m']*(max(0.,self.descent-s['descent_deadband_m'])-old)
            score=0. if h<lo or not m['grasped'] else s['lift_score_at_start']+(s['lift_score_at_max']-s['lift_score_at_start'])*np.clip((h-lo)/(hi-lo),0,1)
            credit('height',score,s['height_reward_budget'])
            picked=m['grasped'] and h>=s['success_min_lift_m'] and m['linear_speed']<=s['hold_linear_speed_m_s'] and m['angular_speed']<=s['hold_angular_speed_rad_s']
            if picked:
                heights=self.hold_heights+[h]
                if max(heights)-min(heights)>s['hold_height_tolerance_m']:heights=[h]
                self.hold_heights=heights;self.hold=len(heights)
            else:self.hold_heights=[];self.hold=0
            credit('hold',self.hold/t['pick_hold_steps'],s['hold_progress_reward_budget'])
            success=self.hold>=t['pick_hold_steps'];phase='hold' if self.hold else ('lift' if m['grasped'] else 'approach')
        else:
            if self.skill=='place':
                controlled=m['grasped'] and local and m['tilt_degrees']<=s['controlled_lowering_tilt_degrees'] and m['linear_speed']<=s['controlled_lowering_max_speed_m_s']
                if controlled:credit('lowering',(s['initial_object_clearance_m']-max(0.,m['clearance_m']))/s['initial_object_clearance_m'],s['lowering_progress_reward_budget'])
                event('contact',contact and m['grasped'],s['stable_contact_event_bonus'])
            else:
                d0=max(self.initial['eef_distance'],1e-6)
                credit('approach',1-m['eef_distance']/d0,s['approach_progress_reward_budget'])
                event('grasp',m['grasped'],s['grasp_event_bonus'])
                event('controlled_lift',m['grasped'] and m['lift_m']>=s['controlled_lift_min_m'],s['controlled_lift_event_bonus'])
                if m['grasped'] or contact:credit('orientation',(self.initial['tilt_degrees']-m['tilt_degrees'])/max(self.initial['tilt_degrees'],1),s['orientation_progress_reward_budget'])
            event('release',contact and released and self.release_eligible,s['conditional_release_event_bonus'])
            event('retreat',stable and 'release' in self.events,s['retreat_event_bonus'])
            credit('stability',self.stable/t['stable_steps'],s['stability_progress_reward_budget'])
            success=self.stable>=t['stable_steps'];phase='stable' if stable else ('retreat' if contact and released else ('contact' if contact else ('manipulate' if m['grasped'] else 'approach')))
        failure=m['object_pos'][2]<m['table_z']-.10;reason='fell_off_table' if failure else None
        if self.skill=='pick' and xy>=s['horizontal_failure_radius_m']:failure=True;reason='pick_xy_boundary'
        if self.skill=='upright' and xy>=s['horizontal_failure_radius_m']:failure=True;reason='upright_xy_boundary'
        success=bool(success and not failure);term=bool(success or failure);trunc=self.step>=c['max_native_steps_per_skill'] and not term
        if success:parts['success']=c['reward']['terminal_bonus'];reason='success'
        if failure:parts['failure']=-c['reward']['failure_penalty']
        if trunc:parts['timeout']=-c['reward']['timeout_penalty'];reason='timeout_200'
        reward=float(sum(parts.values()));self.total+=reward;self.done=term or trunc;self.prev=copy.deepcopy(m)
        info=dict(m,native_step=self.step,remaining_steps=max(0,c['max_native_steps_per_skill']-self.step),max_object_lift_m=self.peak,object_lift_m=m['lift_m'],xy_offset_m=xy,xy_path_m=self.path,hold_steps=self.hold,stable_count=self.stable,pick_count=self.hold,phase=phase,reward_components=parts,reward=reward,episode_reward=self.total,skill_success=success,failure=bool(failure),termination_reason=reason,terminated=term,truncated=bool(trunc),earned=dict(self.earned),events=sorted(self.events),cumulative_descent_m=self.descent)
        return reward,term,bool(trunc),info
