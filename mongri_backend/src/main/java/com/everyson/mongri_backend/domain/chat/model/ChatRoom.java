package com.everyson.mongri_backend.domain.chat.model;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ChatRoom {

    private String roomId;

    public static ChatRoom create(String sid) {
        ChatRoom chatRoom = new ChatRoom();
        chatRoom.roomId = sid;

        return chatRoom;
    }
}
