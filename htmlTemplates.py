css = '''
<style>
.chat-message{
    padding:1.5rem; border-radius:0.5rem;margin_bottom: 1rem; display: flex
}
.chat-message.user{
    background-color: #2b313e
}
.chat-message.bot{
    background-color:#475063
}
.chat-message.bot{
    background-color:#475063
}
.chat-message .avator{
 width:15%;
}
.chat-messsage .avator img{
  max-width: 78px;
  max-height:78px;
  border-radius:50%;
  object-fit:cover;
}
.chat-message .message{
 width:85;
 padding:0 1.5rem;
 color:#fff;
}
'''

bot_template='''
<div class="chat-message bot">
    <div class="avator>
        <img src = "" style="max-height:78px;max_width:78px; border-radius:50%"
    </div>
    <div class="message">{{$MSG}}</div>
</div>
'''

user_template='''
<div class="chat-message user">
    <div class="avator">
        <img src="">
    </div>
    <div class="message">{{$MSG}}</div>
</div>
'''